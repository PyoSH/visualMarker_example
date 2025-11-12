#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory> // for std::shared_ptr

// 기존에 작성한 헤더 파일 포함
#include "../includes/MarkerDetection.h"
#include "../includes/imageProcessor.h"

class ArucoProcessorNode {
public:
    ArucoProcessorNode() : it_(nh_), got_camera_info_(false) {
        // 1. 파라미터 로드 (Launch 파일에서 변경 가능)
        nh_.param<double>("marker_length", marker_length_, 0.168); // 기본값 0.05m (5cm)
        nh_.param<int>("dictionary_id", dictionary_id_, cv::aruco::DICT_4X4_50);

        ROS_INFO("Settings - Marker Length: %.3f m, Dict ID: %d", marker_length_, dictionary_id_);

        // 2. ImageProcessor 파이프라인 설정 (생성 시 1회 설정)
        setupImageProcessor();

        // 3. Subscriber 설정
        // 카메라 정보는 1번만 받으면 되므로 큐 사이즈 1
        cam_info_sub_ = nh_.subscribe("camera_info", 1, &ArucoProcessorNode::cameraInfoCallback, this);

        // 이미지 구독
        image_sub_ = it_.subscribe("image_raw", 1, &ArucoProcessorNode::imageCallback, this);

        ROS_INFO("Waiting for camera info...");
    }

private:
    // --- [핵심 1] ImageProcessor 파이프라인 구성 ---
    void setupImageProcessor() {
        // Step 1: Grayscale
        processor_.addStep([](const cv::Mat& img) -> cv::Mat {
            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            return gray;
        });

        // Step 2: CLAHE (조명 보정)
        processor_.addStep([](const cv::Mat& img) -> cv::Mat {
            cv::Mat enhanced;
            auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(img, enhanced);
            return enhanced;
        });

        // Step 3: Gaussian Blur (노이즈 제거)
        processor_.addStep([](const cv::Mat& img) -> cv::Mat {
            cv::Mat blurred;
            cv::GaussianBlur(img, blurred, cv::Size(5, 5), 0);
            return blurred;
        });

        ROS_INFO("ImageProcessor pipeline configured: Gray -> CLAHE -> Blur");
    }

    // --- [핵심 2] CameraInfo 콜백 함수 ---
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
    {
        if (got_camera_info_) return; // 이미 초기화되었으면 패스

        ROS_INFO("Received Camera Info!");

        // ROS 메시지 -> OpenCV Mat 변환
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
            msg->K[0], msg->K[1], msg->K[2],
            msg->K[3], msg->K[4], msg->K[5],
            msg->K[6], msg->K[7], msg->K[8]);

        cv::Mat distCoeffs = cv::Mat(msg->D, true);

        // MarkerDetection 객체 생성 (이제 카메라 정보를 알았으므로 생성 가능)
        detector_ = std::make_shared<MarkerDetection>(
            cameraMatrix, distCoeffs, marker_length_, dictionary_id_
        );

        got_camera_info_ = true;

        // (선택 사항) 더 이상 카메라 정보가 필요 없으면 구독 취소
        // cam_info_sub_.shutdown();
    }

    // --- [핵심 3] Image 콜백 함수 ---
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

        // 2. 영상 전처리 (ImageProcessor 사용)
        // raw_img는 보존하고, 검출용 이미지를 생성
        cv::Mat processed_img = processor_.process(raw_img);

        // 3. 마커 검출 및 Pose 추정
        ArucoResult results;
        bool found = detector_->estimatePose(processed_img, results);

        // 4. 결과 출력 및 시각화
        if (found) {
            ROS_INFO_THROTTLE(1.0, "Found %lu markers", results.ids.size());

            for (size_t i = 0; i < results.ids.size(); ++i) {
                int id = results.ids[i];
                cv::Vec3d tvec = results.tvecs[i]; // Position
                cv::Vec3d rvec = results.rvecs[i]; // Orientation (Rodrigues)

                // 콘솔 출력 (Position & Orientation)
                printf("[ID: %d] Pos(x,y,z): %.3f, %.3f, %.3f | Rot(rx,ry,rz): %.3f, %.3f, %.3f\n",
                       id,
                       tvec[0], tvec[1], tvec[2],
                       rvec[0], rvec[1], rvec[2]);
            }

            // (선택) 시각화: 원본 이미지에 그리기
            // detector_->drawResults(raw_img, results);
            // cv::imshow("Result", raw_img);

            // (선택) 전처리 과정 확인
            // cv::imshow("Processed Input", processed_img);
            // cv::waitKey(1);
        }
    }

    // ROS 핸들러
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    ros::Subscriber cam_info_sub_;
    image_transport::Subscriber image_sub_;

    // 커스텀 클래스 객체
    std::shared_ptr<MarkerDetection> detector_; // 나중에 초기화하므로 포인터 사용
    ImageProcessor processor_;

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