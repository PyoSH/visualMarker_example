#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "includes/MarkerDetection.h" // Aruco 검출 클래스는 그대로 사용

// --- 1. 전역 변수 선언 ---
std::shared_ptr<MarkerDetection> estimator;

cv::Mat cameraMatrix;
cv::Mat distCoeffs;

double markerLength = 0.05; // 기본값
int dictionaryId = cv::aruco::DICT_6X6_250; // 기본값

bool gotCameraInfo = false;

image_transport::Publisher image_pub;
ros::Subscriber cam_info_sub;

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (gotCameraInfo) {
        return;
    }

    ROS_INFO("Received camera info!");

    cameraMatrix = (cv::Mat_<double>(3, 3) <<
        msg->K[0], msg->K[1], msg->K[2],
        msg->K[3], msg->K[4], msg->K[5],
        msg->K[6], msg->K[7], msg->K[8]);

    distCoeffs = cv::Mat(msg->D, true);

    estimator = std::make_shared<MarkerDetection>(
        cameraMatrix, distCoeffs, markerLength, dictionaryId
    );

    gotCameraInfo = true; // 플래그 설정

    cam_info_sub.shutdown();
    ROS_INFO("Camera info processed. Marker detection is active.");
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg) {

    if (!gotCameraInfo || !estimator) {
        ROS_WARN_THROTTLE(1.0, "Waiting for camera info...");
        return;
    }

    // ROS 이미지를 OpenCV 이미지(cv::Mat)로 변환
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // 마커 검출 및 Pose 추정
    ArucoResult results;
    bool found = estimator->estimatePose(cv_ptr->image, results);

    if (found) {
        estimator->drawResults(cv_ptr->image, results);

        std::cout << "--- Found " << results.ids.size() << " Markers ---" << std::endl;
        for (size_t i = 0; i < results.ids.size(); ++i) {
            int currentId = results.ids[i];
            cv::Vec3d rvec = results.rvecs[i];
            cv::Vec3d tvec = results.tvecs[i];

            std::cout << "ID: " << currentId
                      << " | tvec: [" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << "]"
                      << " | rvec: [" << rvec[0] << ", " << rvec[1] << ", " << rvec[2] << "]"
                      << std::endl;
        }
    } else {
         ROS_INFO_THROTTLE(1.0, "No markers found in this frame.");
    }

    image_pub.publish(cv_ptr->toImageMsg());
}


// --- 4. Main 함수 ---
int main(int argc, char** argv) {
    ros::init(argc, argv, "aruco_detector_simple_node");

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    // --- ROS 파라미터 불러오기 (전역 변수에 저장) ---
    nh.param<double>("marker_length", markerLength, 0.05);
    nh.param<int>("dictionary_id", dictionaryId, cv::aruco::DICT_6X6_250);

    ROS_INFO("Using marker length: %f m", markerLength);
    ROS_INFO("Using dictionary ID: %d", dictionaryId);

    // --- Publisher 및 Subscriber 설정 ---

    // 결과 이미지 발행
    image_pub = it.advertise("marker_debug_image", 1);

    // 카메라 정보 구독 (콜백: cameraInfoCallback)
    cam_info_sub = nh.subscribe("camera_info", 1, cameraInfoCallback);

    // 원본 이미지 구독 (콜백: imageCallback)
    image_transport::Subscriber image_sub = it.subscribe("image", 1, imageCallback);

    ROS_INFO("Aruco simple node started. Waiting for camera info...");

    ros::spin(); // 콜백 대기

    return 0;
}