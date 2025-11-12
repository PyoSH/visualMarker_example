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
    ArucoProcessorNode() : it_(nh_), got_camera_info_(true) {  // !!!!!!!!
        // 1. 파라미터 로드 (Launch 파일에서 변경 가능)
        nh_.param<double>("marker_length", marker_length_, 0.168); // 기본값 0.05m (5cm)
        nh_.param<int>("dictionary_id", dictionary_id_, cv::aruco::DICT_4X4_50);

        ROS_INFO("Settings - Marker Length: %.3f m, Dict ID: %d", marker_length_, dictionary_id_);

        // 3. Subscriber 설정
        // 카메라 정보는 1번만 받으면 되므로 큐 사이즈 1
        cam_info_sub_ = nh_.subscribe("camera_info", 1, &ArucoProcessorNode::cameraInfoCallback, this);

        // 이미지 구독
        image_sub_ = it_.subscribe("/aruco_marker_noise_img_PSH", 1, &ArucoProcessorNode::imageCallback, this);

        ROS_INFO("Waiting for camera info...");

        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
            1803.10147,     0.0,        980.60876,  // fx, 0, cx
            0.0,            1802.02133, 650.23663,  // 0, fy, cy
            0.0,            0.0,        1.0);
        
        cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.388341, 0.183527, -0.000277, 0.001198, 0.0); // k1, k2, p1, p2, k3

        detector_ = std::make_shared<MarkerDetection>(
            cameraMatrix, distCoeffs, marker_length_, dictionary_id_
        );
    }

private:
    // --- [핵심 1] ImageProcessor 파이프라인 구성 ---
    void setupImageProcessor(bool flag_lighting, bool flag_Blur, bool flag_something) {
        // Step 1: Grayscale
        processor_.addStep([](const cv::Mat& img) -> cv::Mat {
            cv::Mat gray;
            if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            else gray = img.clone();
            
            return gray;
        });

        if (flag_lighting){
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
        }
        
        if (flag_Blur){
            // [블러 제거] 언샤프 마스킹 (Unsharp Masking)
            processor_.addStep([](const cv::Mat& img) -> cv::Mat {
                cv::Mat blurred, sharpened;
                
                // 1. 이미지의 저주파 성분(뭉개진 부분)을 추출
                // sigma 값을 조절하여 샤프닝의 '범위'를 결정 (값이 클수록 굵은 선이 강조됨)
                cv::GaussianBlur(img, blurred, cv::Size(0, 0), 3.0);
                
                // 2. 원본과 블러된 이미지의 차이를 이용해 엣지 강조
                // addWeighted(src1, alpha, src2, beta, gamma, dst)
                // 식: Result = img * 1.5 + blurred * (-0.5)
                // 즉, 원본을 1.5배 강조하고, 뭉개진 부분을 0.5배 뺍니다.
                cv::addWeighted(img, 2.5, blurred, -1.5, 0, sharpened);
                
                return sharpened;
            });

            processor_.addStep([](const cv::Mat& img) -> cv::Mat {
                cv::Mat binary;
                // THRESH_BINARY | THRESH_OTSU 사용
                // 임계값(0)은 무시되고, Otsu 알고리즘이 최적의 값을 자동으로 계산합니다.
                // 결과: 배경은 255(흰색), 마커는 0(검은색)으로 '덩어리'째 분리됨
                cv::threshold(img, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                
                // 만약 조명 때문에 노이즈가 좀 있다면, 모폴로지 열기(Opening)로 점들을 제거
                // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                // cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
                
                return binary;
            });
        }
        
        std::cout <<"ImageProcessor pipeline: " << processor_.m_steps.size() << std::endl;
    }

    // --- [핵심 2] CameraInfo 콜백 함수 ---
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
    {
        if (got_camera_info_) return; // 이미 초기화되었으면 패스

        ROS_INFO("Received Camera Info!");

        // ROS 메시지 -> OpenCV Mat 변환
        // cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        //     msg->K[0], msg->K[1], msg->K[2],
        //     msg->K[3], msg->K[4], msg->K[5],
        //     msg->K[6], msg->K[7], msg->K[8]);
        // cv::Mat distCoeffs = cv::Mat(msg->D, true);

        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
            1803.10147,     0.0,        980.60876,  // fx, 0, cx
            0.0,            1802.02133, 650.23663,  // 0, fy, cy
            0.0,            0.0,        1.0);
        
        cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.388341, 0.183527, -0.000277, 0.001198, 0.0); // k1, k2, p1, p2, k3

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

        // 2.1 ImageProcessor 파이프라인 설정 (생성 시 1회 설정)
        setupImageProcessor(false, false, false);
        std::vector<cv::Mat> debug_imgs;
        cv::Mat processed_img = processor_.process(raw_img, debug_imgs);

        publishDebugImages(debug_imgs, msg->header);

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
            detector_->drawResults(raw_img, results);
            cv::Mat img_small;
            cv::resize(raw_img, img_small, cv::Size(640, 480));
            cv::imshow("Result", img_small);

            // (선택) 전처리 과정 확인
            // cv::imshow("Processed Input", processed_img);
            cv::waitKey(1);

        }else{
            std::cout <<"No marker found!" << std::endl;
            /*
            여기에 왜 검출 안됐는지 분석하는 코드 넣고, 다시 검출하도록 하기.
            */
        }
        processor_.clearSteps(); // 반복될 때마다 스텝이 누적되지 않도록.

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

    // ROS 핸들러
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    ros::Subscriber cam_info_sub_;
    image_transport::Subscriber image_sub_;

    // [추가] 디버그용 Publisher 리스트
    std::vector<image_transport::Publisher> debug_pubs_;

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