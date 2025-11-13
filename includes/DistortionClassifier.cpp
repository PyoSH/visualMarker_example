#include "DistortionClassifier.h"

DistortionClassifier::DistortionClassifier() {
    // 초기화가 필요하다면 여기에 작성
}

DistortionType DistortionClassifier::classify(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img.clone();

    // 판별 우선순위: Occlusion(가장 심각) -> Noise -> Blur -> Clean
    if (isOccluded(gray)) return DistortionType::OCCLUSION;
    if (isSaltAndPepper(gray)) return DistortionType::SALT_PEPPER;
    if (isBlurred(gray)) return DistortionType::BLUR;

    return DistortionType::CLEAN;
}

bool DistortionClassifier::isBlurred(const cv::Mat& img) {
    cv::Mat laplacian;
    cv::Laplacian(img, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    double variance = stddev.val[0] * stddev.val[0];
    return variance < 150.0; // 임계값 튜닝 필요
}

bool DistortionClassifier::isSaltAndPepper(const cv::Mat& img) {
    cv::Mat median;
    cv::medianBlur(img, median, 3); // 3x3
    
    cv::Mat diff;
    cv::absdiff(img, median, diff);
    std::cout << cv::mean(diff)[0] << std::endl;
    
    return cv::mean(diff)[0] > 4.0; // 임계값 튜닝 필요
}

bool DistortionClassifier::isOccluded(const cv::Mat& img) {
    cv::Mat binary;
    // 1. 반전 이진화 (검은 물체 -> 흰색 덩어리)
    cv::threshold(img, binary, 50, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    // 2. 윤곽선 및 계층 구조 검출
    cv::findContours(binary, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    double imgArea = img.rows * img.cols;

    for (size_t i = 0; i < contours.size(); i++) {
        // 노이즈 무시
        if (cv::contourArea(contours[i]) < imgArea * 0.03) continue;

        // 3. 다각형 근사 (꼭짓점 개수 파악)
        std::vector<cv::Point> approx;
        double epsilon = 0.02 * cv::arcLength(contours[i], true); // 오차율을 조금 줄여서 디테일하게 봅니다
        cv::approxPolyDP(contours[i], approx, epsilon, true);

        bool isConvex = cv::isContourConvex(approx);
        int vertices = approx.size();
        int child_idx = hierarchy[i][2]; // 첫 번째 자식 인덱스
        bool hasChildren = (child_idx != -1);

        // --- [Case 1: 단순 가림막 (Isolated Occlusion)] ---
        // 사각형이고(4) + 볼록하고(Convex) + 내부에 아무것도 없음(No Children)
        if (vertices == 4 && isConvex && !hasChildren) {
            return true; // "검은색 색종이" 발견
        }

        // --- [Case 2: 겹쳐진 마커 (Merged/Touching Occlusion)] ---
        // 이것이 바로 사용자님이 말씀하신 "두 사각형이 겹친 경우"입니다.
        // 특징: 내부에 마커 패턴은 있는데(Has Children), 겉모양이 사각형이 아님(Not 4 Vertices OR Concave)
        if (hasChildren) {
            if (vertices > 4 || !isConvex) {
                // "내용물은 마커 같은데, 껍데기가 찌그러져 있거나 뿔이 났다" -> 가림막이 붙음!
                return true; 
            }
        }
        
        // (참고: vertices==4 && isConvex && hasChildren 인 경우는 '정상 마커' 후보이므로 넘어감)
    }
    
    return false;
}