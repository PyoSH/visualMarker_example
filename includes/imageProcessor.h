#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <functional> // std::function을 사용하기 위해 필수

/**
 * @typedef ProcessingStep
 * @brief cv::Mat을 입력받아 cv::Mat을 반환하는 모든 함수를 나타내는 타입
 */
using ProcessingStep = std::function<cv::Mat(const cv::Mat&)>;

/**
 * @class ImageProcessor
 * @brief 영상 처리 단계를 동적으로 추가하고 순차적으로 실행하는 파이프라인 클래스
 */
class ImageProcessor {
public:
    ImageProcessor() = default; // 기본 생성자

    /**
     * @brief 파이프라인에 처리 단계를 추가합니다.
     * @param step ProcessingStep 타입의 함수 (람다, 일반 함수 등)
     */
    void addStep(const ProcessingStep& step);

    /**
     * @brief 입력 이미지를 파이프라인의 모든 단계에 순차적으로 적용합니다.
     * @param inputImage 원본 이미지
     * @return 모든 처리가 완료된 최종 이미지
     */
    cv::Mat process(const cv::Mat& inputImage, std::vector<cv::Mat>& debugImages);

    /**
     * @brief 파이프라인에 등록된 모든 처리 단계를 초기화합니다.
     */
    void clearSteps();

    std::vector<ProcessingStep> m_steps;

private:
    // 처리 단계들을 순서대로 저장하는 벡터
    // std::vector<ProcessingStep> m_steps;
};

#endif //IMAGE_PROCESSOR_H