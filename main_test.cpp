#include <opencv2/opencv.hpp>
#include "includes/MarkerDetection.h"
#include "includes/imageProcessor.h"
#include <iostream>

cv::Mat applyClahe(const cv::Mat& input) {
    if (input.channels() != 1) {
        std::cerr << "CLAHE는 1채널 이미지에만 적용됩니다." << std::endl;
        return input;
    }
    cv::Mat claheOutput;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(input, claheOutput);
    return claheOutput;
}

int main() {
    // !!! 이 값들은 반드시 실제 카메라 보정을 통해 얻은 값을 사용해야 함 !!!
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        600.0, 0.0,   320.0,  // fx, 0, cx
        0.0,   600.0, 240.0,  // 0, fy, cy
        0.0,   0.0,   1.0);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.1, -0.05, 0.0, 0.0, 0.0); // k1, k2, p1, p2, k3

    // 마커의 실제 한 변 길이 (미터 단위)
    float markerLength = 0.05; // 5cm

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    MarkerDetection estimator(cameraMatrix, distCoeffs, markerLength);


    cv::Mat image_test;
    std::string imagePath = "../test_imgs/markers_desk.jpg";
    image_test = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image_test.empty()) {
        std::cerr << "오류: 이미지를 로드할 수 없습니다: " << imagePath << std::endl;
        return -1; // 프로그램 종료 또는 에러 처리
    }

    ImageProcessor processor;

    // 단계 1: 그레이스케일 변환 - 람다 함수 형식
    processor.addStep([](const cv::Mat& img) -> cv::Mat {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        return gray;
    });

    // (Color -> GRAY)
    cv::Mat finalImage = processor.process(image_test);
    cv::imshow("Original Image", image_test);
    cv::imshow("Final processed Image", finalImage);

    ArucoResult results; // 결과를 저장할 구조체

    cv::Mat processImage = finalImage.clone();
    bool found = estimator.estimatePose(processImage, results);

    std::cout << found << std::endl;

    if (found) {
        estimator.drawResults(processImage, results);

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
        std::cout << "마커는 찾았으나 Pose 추정에 실패했습니다." << std::endl;
    }

    cv::imshow("Aruco Pose Estimation", processImage);

    cv::waitKey(0);

    return 0;
}