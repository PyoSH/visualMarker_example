#ifndef DISTORTION_CLASSIFIER_H
#define DISTORTION_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <numeric>

// 다른 클래스(main 등)에서 쓰기 편하게 클래스 밖 전역으로 선언
enum class DistortionType {
    CLEAN,          // 정상
    BLUR,           // 흐림 (초점 나감)
    SALT_PEPPER,    // 소금-후추 잡음
    OCCLUSION       // 가림/끊김 (검은 사각형 등)
};

class DistortionClassifier {
public:
    DistortionClassifier(); // 생성자

    // 메인 분류 함수
    DistortionType classify(const cv::Mat& img);

private:
    // 내부 판별 로직들
    bool isBlurred(const cv::Mat& img);
    bool isSaltAndPepper(const cv::Mat& img);
    bool isOccluded(const cv::Mat& img); // 가림 확인 (이전 턴의 사각형 로직)
};

#endif // DISTORTION_CLASSIFIER_H