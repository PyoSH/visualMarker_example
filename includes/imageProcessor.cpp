#include "imageProcessor.h"

void ImageProcessor::addStep(const ProcessingStep& step) {
    m_steps.push_back(step);
}

void ImageProcessor::clearSteps() {
    m_steps.clear();
}

cv::Mat ImageProcessor::process(const cv::Mat& inputImage, std::vector<cv::Mat>& debugImages) {
    cv::Mat currentImage = inputImage.clone();
    debugImages.clear(); // 벡터 초기화

    // 0단계: 원본 저장 (선택 사항)
    // debugImages.push_back(currentImage.clone());

    for (const auto& stepFunction : m_steps) {
        currentImage = stepFunction(currentImage);
        
        // [핵심] 현재 단계의 결과를 디버그 벡터에 복사하여 저장
        debugImages.push_back(currentImage.clone());
    }

    return currentImage;
}