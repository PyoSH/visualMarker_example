#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Float64MultiArray.h> // [필수] 결과 전송용
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory> // for std::shared_ptr

// 기존에 작성한 헤더 파일 포함
#include "../includes/MarkerDetection.h"
#include "../includes/imageProcessor.h"
#include "../includes/DistortionClassifier.h"

std::string getDistortionName(DistortionType type) {
    switch (type) {
        case DistortionType::CLEAN:       return "CLEAN";
        case DistortionType::BLUR:        return "BLUR";
        case DistortionType::SALT_PEPPER: return "SALT_PEPPER";
        case DistortionType::OCCLUSION:   return "OCCLUSION";
        default:                          return "UNKNOWN";
    }
}

class ArucoProcessorNode {
public:
    ArucoProcessorNode() : it_(nh_), got_camera_info_(true) {  // !!!!!!!!
        // 1. 파라미터 로드 (Launch 파일에서 변경 가능)
        nh_.param<double>("marker_length", marker_length_, 0.168); // 기본값 0.05m (5cm)
        nh_.param<int>("dictionary_id", dictionary_id_, cv::aruco::DICT_4X4_50);

        ROS_INFO("Settings - Marker Length: %.3f m, Dict ID: %d", marker_length_, dictionary_id_);

        // 3. Subscriber 설정
        // 카메라 정보는 1번만 받으면 되므로 큐 사이즈 1
        cam_info_sub_ = nh_.subscribe("camera_info", 1, &ArucoProcessorNode::cameraInfoCallback, this);

        // 이미지 구독
        image_sub_ = it_.subscribe("/pub_filtered_imgs", 1, &ArucoProcessorNode::imageCallback, this);

        // Pub: Master PC로 보내는 결과 (Float64MultiArray)
        pose_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/return_pose_info", 1);

        ROS_INFO("Waiting for camera info...");

        cameraMatrix = (cv::Mat_<double>(3, 3) <<
            1975.8198,     0.0,        958.0988,  // fx, 0, cx
            0.0,           1976.17418, 587.56144,  // 0, fy, cy
            0.0,           0.0,        1.0);
        
        distCoeffs = (cv::Mat_<double>(1, 5) << -0.382539, 0.188592, 0.001473, -0.000317, 0.0); // k1, k2, p1, p2, k3

        detector_ = std::make_shared<MarkerDetection>(
            cameraMatrix, distCoeffs, marker_length_, dictionary_id_
        );
    }

private:
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
    {
        if (got_camera_info_) return; // 이미 초기화되었으면 패스

        ROS_INFO("Received Camera Info!");

        // ROS 메시지 -> OpenCV Mat 변환
        cameraMatrix = (cv::Mat_<double>(3, 3) <<
            msg->K[0], msg->K[1], msg->K[2],
            msg->K[3], msg->K[4], msg->K[5],
            msg->K[6], msg->K[7], msg->K[8]);
        distCoeffs = cv::Mat(msg->D, true);

        // MarkerDetection 객체 생성 (이제 카메라 정보를 알았으므로 생성 가능)
        detector_ = std::make_shared<MarkerDetection>(
            cameraMatrix, distCoeffs, marker_length_, dictionary_id_
        );

        got_camera_info_ = true;

        // (선택 사항) 더 이상 카메라 정보가 필요 없으면 구독 취소
        cam_info_sub_.shutdown();
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        if (!got_camera_info_ || !detector_) {
            ROS_WARN_THROTTLE(2.0, "Waiting for camera intrinsics...");
            return;
        }

        // 1. ROS Image -> cv::Mat 변환
        cv::Mat raw_img;
        try {
            raw_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        std::vector<cv::Mat> debug_imgs;

        std::cout << "[1단계] 입력된 이미지에서 마커 검출하기" << std::endl;

        ArucoResult results;
        bool found = detector_->estimatePose(raw_img, results);
        bool success = false;

        // 4. 결과 출력 및 시각화
        float error_thres = 3.0;

        if (found) {
            ROS_INFO_THROTTLE(1.0, "Found %lu markers", results.ids.size());
            double error = getReprojectionError(results.corners[0], results.rvecs[0], results.tvecs[0], cameraMatrix, distCoeffs, marker_length_);

            if (error < error_thres){
                success = true;
                std::cout << "[1단계 성공] 원본 검출 완료 (Error: " << error << ")" << std::endl;
            }else{
                std::cout << "[1단계 불안정] 신뢰도 낮음 (Error: " << error << "). 재검출 필요." << std::endl;
            }
        }
        cv::Mat processed_img = raw_img.clone();

        if (!success){
            std::cout << "[2단계] 마커 검출 실패, 원인 분류" << std::endl;

            DistortionType type = classifier.classify(raw_img);
            std::cout << " -> 감지된 원인: " << getDistortionName(type) << std::endl;

            processor_.configureForType(type);
            processed_img = processor_.process(raw_img,debug_imgs);

            bool found_2nd = detector_->estimatePose(processed_img, results);

            if (found_2nd){
                success = true;
                std::cout << "[3단계 성공] 전처리 후 검출 완료!" << std::endl;
            }else{
                std::cout << "[Fail] 전처리 후에도 마커 검출 실패." << std::endl;
            }
        }

        if (!raw_img.empty() && !processed_img.empty()) {
    
            cv::Mat left_img = raw_img.clone();       // 원본 (왼쪽)
            cv::Mat right_img;                        // 처리된 이미지 (오른쪽)

            // 1. 채널 맞추기 (흑백 -> 컬러 포맷으로 변환)
            // hconcat은 채널 수가 다르면 에러가 납니다.
            if (processed_img.channels() == 1) {
                cv::cvtColor(processed_img, right_img, cv::COLOR_GRAY2BGR);
            } else {
                right_img = processed_img.clone();
            }

            // (선택) 만약 결과(Draw된 것)를 오른쪽에 보고 싶다면 아래 줄 주석 해제
            // right_img = debug_imgs.back(); // drawResults가 적용된 마지막 이미지

            // 2. 텍스트 추가 (구분을 위해)
            cv::putText(left_img, "Raw Input", cv::Point(30, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(right_img, "Processed", cv::Point(30, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

            // 3. 두 이미지 가로로 합치기 (Horizontal Concatenate)
            cv::Mat comparison;
            cv::hconcat(left_img, right_img, comparison);

            // 4. 윈도우 띄우기 (Slave PC 요구사항)
            // 창 크기가 너무 크면 줄여서 보여주기 (선택 사항)
            cv::resize(comparison, comparison, cv::Size(), 0.5, 0.5); 
            
            cv::imshow("Monitor: Raw vs Processed", comparison);
            cv::waitKey(1); // ROS에서는 spin()이 돌지만 imshow 갱신을 위해 필수
        }
        
        if (success){
            for (size_t i = 0; i < results.ids.size(); ++i) {
                int id = results.ids[i];
                cv::Vec3d tvec = results.tvecs[i]; // Position
                cv::Vec3d rvec = results.rvecs[i]; // Orientation (Rodrigues)

                cv::Vec3d euler = getEulerAngles(rvec); // 오일러 각

                // 콘솔 출력 (Position & Orientation)
                printf("[ID: %d] Pos(x,y,z): %.3f, %.3f, %.3f | Rot(rx,ry,rz): %.3f, %.3f, %.3f\n",
                    id,
                    tvec[0], tvec[1], tvec[2],
                    euler[0], euler[1], euler[2]);
                
                /*
                // [추가] ROS 메시지 발행 (Master PC 요구사항)
                // 토픽명: /return_pose_info, 타입: Float64MultiArray
                // 데이터 순서: x, y, z, roll, pitch, yaw
                */
                std_msgs::Float64MultiArray pose_msg;
                pose_msg.data.resize(6);
                pose_msg.data[0] = tvec[0];
                pose_msg.data[1] = tvec[1];
                pose_msg.data[2] = tvec[2];
                pose_msg.data[3] = euler[0]; // Roll
                pose_msg.data[4] = euler[1]; // Pitch
                pose_msg.data[5] = euler[2]; // Yaw
                pose_pub_.publish(pose_msg);
            }

            // (선택) 시각화: 원본 이미지에 그리기
            detector_->drawResults(raw_img, results);
            debug_imgs.push_back(raw_img.clone());
        }else{
            std::cout << " ?? IDK ?? " << std::endl;
        }
        publishDebugImages(debug_imgs, msg->header);
    }

    // --- [핵심 기능] 디버그 이미지 발행 함수 ---
    void publishDebugImages(const std::vector<cv::Mat>& images, std_msgs::Header header) {
        // 1. Publisher가 부족하면 추가 생성 (Lazy Initialization)
        while (debug_pubs_.size() < images.size()) {
            int id = debug_pubs_.size();
            // 토픽 이름 예: /debug/step_0, /debug/step_1 ...
            std::string topic_name = "debug/step_" + std::to_string(id);
            debug_pubs_.push_back(it_.advertise(topic_name, 1));
            ROS_INFO("Created debug publisher: %s", topic_name.c_str());
        }

        // 2. 각 단계별 이미지 발행
        for (size_t i = 0; i < images.size(); ++i) {
            if (images[i].empty()) continue;

            // 인코딩 결정 (흑백 vs 컬러)
            std::string encoding;
            if (images[i].channels() == 1) encoding = sensor_msgs::image_encodings::MONO8;
            else encoding = sensor_msgs::image_encodings::BGR8;

            // cv_bridge 메시지 생성 및 발행
            sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(header, encoding, images[i]).toImageMsg();
            debug_pubs_[i].publish(out_msg);
        }
    }

    double getReprojectionError(const std::vector<cv::Point2f>& detectedCorners,
                                const cv::Vec3d& rvec,
                                const cv::Vec3d& tvec,
                                const cv::Mat& cameraMatrix,
                                const cv::Mat& distCoeffs,
                                float markerLength) 
    {        
        // 1. 마커의 3D 좌표계 정의 (중심이 0,0,0인 평면 사각형)
        std::vector<cv::Point3f> objectPoints;
        float halfSize = markerLength / 2.0f;
        // 순서: Top-Left, Top-Right, Bottom-Right, Bottom-Left (OpenCV ArUco 표준)
        objectPoints.push_back(cv::Point3f(-halfSize, halfSize, 0));
        objectPoints.push_back(cv::Point3f(halfSize, halfSize, 0));
        objectPoints.push_back(cv::Point3f(halfSize, -halfSize, 0));
        objectPoints.push_back(cv::Point3f(-halfSize, -halfSize, 0));

        // 2. 3D 점을 현재 Pose(rvec, tvec)를 이용해 2D 화면으로 투영
        std::vector<cv::Point2f> projectedPoints;
        cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

        // 3. 검출된 좌표와 투영된 좌표 사이의 거리(오차) 계산
        double totalError = 0.0;
        for (size_t i = 0; i < detectedCorners.size(); ++i) {
            // 유클리드 거리 제곱 합
            double error = cv::norm(detectedCorners[i] - projectedPoints[i]); 
            totalError += error * error;
        }

        // RMS (Root Mean Square) 에러 반환
        return std::sqrt(totalError / detectedCorners.size());
    }

    cv::Vec3d getEulerAngles(cv::Vec3d rvec) {
        cv::Mat R;
        cv::Rodrigues(rvec, R); // 회전 벡터 -> 회전 행렬 변환

        // 회전 행렬에서 오일러 각 추출 (일반적인 Z-Y-X 순서)
        double sy = std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) +
                              R.at<double>(1,0) * R.at<double>(1,0));

        bool singular = sy < 1e-6; // 짐벌락 체크

        double x, y, z;
        if (!singular) {
            x = atan2(R.at<double>(2,1), R.at<double>(2,2));
            y = atan2(-R.at<double>(2,0), sy);
            z = atan2(R.at<double>(1,0), R.at<double>(0,0));
        } else {
            x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
            y = atan2(-R.at<double>(2,0), sy);
            z = 0;
        }

        // Radian -> Degree 변환하여 반환
        return cv::Vec3d(x, y, z) * (180.0 / CV_PI);
    }

    // ROS 핸들러
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    ros::Subscriber cam_info_sub_;
    image_transport::Subscriber image_sub_;
    ros::Publisher pose_pub_;

    // [추가] 디버그용 Publisher 리스트
    std::vector<image_transport::Publisher> debug_pubs_;

    cv::Mat distCoeffs;
    cv::Mat cameraMatrix;

    // 커스텀 클래스 객체
    std::shared_ptr<MarkerDetection> detector_; // 나중에 초기화하므로 포인터 사용
    ImageProcessor processor_;
    DistortionClassifier classifier;

    // 상태 변수 및 파라미터
    bool got_camera_info_;
    double marker_length_;
    int dictionary_id_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "aruco_processor_node");
    ArucoProcessorNode node;
    ros::spin();
    return 0;
}