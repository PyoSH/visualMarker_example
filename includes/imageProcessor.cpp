#include "imageProcessor.h"

void ImageProcessor::addStep(const ProcessingStep& step) {
    m_steps.push_back(step);
}

void ImageProcessor::clearSteps() {
    m_steps.clear();
}

cv::Mat ImageProcessor::process(const cv::Mat& inputImage) {
    // 원본 이미지를 수정하지 않도록 복사본으로 시작
    cv::Mat currentImage = inputImage.clone();

    // m_steps 벡터에 저장된 모든 함수를 순서대로 실행
    for (const auto& stepFunction : m_steps) {
        // 현재 단계의 함수를 실행
        // 이전 단계의 출력이 현재 단계의 입력이 됨
        currentImage = stepFunction(currentImage);
    }

    // 모든 단계를 거친 최종 이미지를 반환
    return currentImage;
}