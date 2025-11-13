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

// ImageProcessor 클래스 내부에 추가
void ImageProcessor::configureForType(DistortionType type) {
    this->clearSteps(); // 기존 스텝 초기화

    // 공통: 그레이스케일 (필요시) & 미디안 필터로 잡음 제거
    this->addStep([](const cv::Mat& img) -> cv::Mat {
        cv::Mat gray;
        if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else gray = img.clone();
        return gray;
    });
    this->addStep([](const cv::Mat& img) -> cv::Mat {
        cv::Mat denoised;
        cv::medianBlur(img, denoised, 5); // 5x5 미디언
        return denoised;
    });

    switch (type) {
        case DistortionType::BLUR:
            // 전략: 강력한 샤프닝 + Otsu 이진화 (속 채우기)
            this->addStep([](const cv::Mat& img) -> cv::Mat {
                cv::Mat sharpened;
                cv::GaussianBlur(img, sharpened, cv::Size(0, 0), 5.0);
                cv::addWeighted(img, 3.0, sharpened, -2.0, 0, sharpened);
                return sharpened;
            });
            this->addStep([](const cv::Mat& img) -> cv::Mat {
                cv::Mat binary;
                cv::threshold(img, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                return binary;
            });
            this->addStep([](const cv::Mat& img) -> cv::Mat {
                cv::Mat bordered;
                
                // 1. 상하좌우에 10픽셀씩 흰색(255) 테두리 추가
                // 결과 이미지는 원본보다 가로/세로 20픽셀씩 커짐
                int border = 10; 
                cv::copyMakeBorder(img, bordered, border, border, border, border, 
                                cv::BORDER_CONSTANT, cv::Scalar(255));
                
                // 2. 원본 크기로 다시 복귀 (Resize)
                // 이미지가 커진 상태로 두면 카메라 행렬(Principal Point)과 안 맞아서 Pose 오차 발생
                // 다시 줄이면서 마커가 중앙으로 살짝 모이는 효과 -> 가장자리 문제 해결
                cv::Mat result;
                cv::resize(bordered, result, img.size(), 0, 0, cv::INTER_NEAREST);
                // *주의: 이진화된 이미지이므로 보간법은 INTER_NEAREST 추천 (엣지 유지)
                
                return result;
            });
            break;

        case DistortionType::SALT_PEPPER:
            // 전략: 미디언 블러 (노이즈 제거) + 기본 적응형 이진화
            this->addStep([](const cv::Mat& img) -> cv::Mat {
                cv::Mat denoised;
                cv::medianBlur(img, denoised, 5); // 5x5 미디언
                return denoised;
            });
            this->addStep([](const cv::Mat& img) -> cv::Mat {
                cv::Mat binary;
                cv::threshold(img, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                return binary;
            });
            break;
        case DistortionType::OCCLUSION:
            // [중요] 람다 캡처에 'this' 포함 (getIntersection 사용)
            this->addStep([this](const cv::Mat& img) -> cv::Mat {
                
                bool DEBUG_MODE = true; 

                // 1. 베이스 이미지 (복사해올 원본 데이터)
                cv::Mat base_binary;
                cv::threshold(img, base_binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

                // 2. 윤곽선 검출용 엣지
                cv::Mat edges;
                cv::Canny(img, edges, 50, 150);
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                cv::dilate(edges, edges, kernel);

                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // 3. [Target] 새로운 흰색 도화지 생성 (결과물)
                cv::Mat clean_result(img.size(), CV_8UC1, cv::Scalar(255)); 
                
                cv::Mat debug_vis;
                if (DEBUG_MODE) cv::cvtColor(base_binary, debug_vis, cv::COLOR_GRAY2BGR);

                // 4. 가장 큰 윤곽선 찾기
                int max_idx = -1;
                double max_area = 0;
                for (size_t i = 0; i < contours.size(); i++) {
                    double area = cv::contourArea(contours[i]);
                    if (area > max_area) {
                        max_area = area;
                        max_idx = (int)i;
                    }
                }

                if (max_idx != -1 && max_area > 1000) {
                    std::vector<cv::Point> approx;
                    double epsilon = 0.02 * cv::arcLength(contours[max_idx], true);
                    cv::approxPolyDP(contours[max_idx], approx, epsilon, true);

                    // 변이 4개보다 많으면 (가려짐/찌그러짐) -> 복원 시도
                    if (approx.size() > 4) {
                        
                        // A. 변 길이 저장
                        struct Edge { double length; cv::Point p1, p2; };
                        std::vector<Edge> edges_vec;

                        for (size_t j = 0; j < approx.size(); ++j) {
                            cv::Point p1 = approx[j];
                            cv::Point p2 = approx[(j + 1) % approx.size()];
                            double len = cv::norm(p1 - p2);
                            edges_vec.push_back({len, p1, p2});
                            
                            if (DEBUG_MODE) cv::line(debug_vis, p1, p2, cv::Scalar(255, 0, 0), 1);
                        }

                        // B. 길이순 정렬 (Top 4 선택)
                        std::sort(edges_vec.begin(), edges_vec.end(), 
                                [](const Edge& a, const Edge& b) { return a.length > b.length; });

                        if (DEBUG_MODE) {
                            for(int m=0; m<4 && m<edges_vec.size(); ++m) 
                                cv::line(debug_vis, edges_vec[m].p1, edges_vec[m].p2, cv::Scalar(0, 255, 0), 2);
                        }

                        // C. 교차점 계산
                        std::vector<cv::Point> corners;
                        for (int j = 0; j < 4; ++j) {
                            for (int k = j + 1; k < 4; ++k) {
                                cv::Point2f pt = this->getIntersection(edges_vec[j].p1, edges_vec[j].p2,
                                                                    edges_vec[k].p1, edges_vec[k].p2);
                                
                                // 유효 범위 체크
                                if (pt.x >= -img.cols && pt.y >= -img.rows && 
                                    pt.x < img.cols*2 && pt.y < img.rows*2) {
                                    
                                    bool duplicate = false;
                                    for(const auto& exist_pt : corners) {
                                        if(cv::norm(exist_pt - (cv::Point)pt) < 20.0) duplicate = true;
                                    }
                                    if(!duplicate) {
                                        corners.push_back(pt);
                                        if (DEBUG_MODE) cv::circle(debug_vis, pt, 5, cv::Scalar(0, 0, 255), -1);
                                    }
                                }
                            }
                        }

                        // D. [복사 로직] 4개의 점 내부만 흰 도화지로 가져오기
                        if (corners.size() == 4) {
                            // 1. 점 순서 정렬 (Convex Hull)
                            std::vector<cv::Point> hull_points;
                            cv::convexHull(corners, hull_points); 

                            // 2. 마스크 생성 (검은 배경)
                            cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

                            // 3. 마스크에 '흰색 다각형' 채우기
                            // (hull_points 내부 영역을 255로 칠함 -> 여기가 복사될 영역)
                            const cv::Point* p = &hull_points[0];
                            int n = (int)hull_points.size();
                            cv::fillConvexPoly(mask, hull_points, cv::Scalar(255));

                            // 4. [Copy] 원본에서 마스크 영역만 오려서 새 도화지에 붙임
                            // mask가 흰색인 부분: base_binary의 픽셀이 clean_result로 복사됨
                            // mask가 검은색인 부분: clean_result의 원래 색(흰색) 유지됨
                            base_binary.copyTo(clean_result, mask);

                            if (DEBUG_MODE) {
                                cv::polylines(debug_vis, &p, &n, 1, true, cv::Scalar(0, 255, 255), 2);
                            }
                        }
                    }
                }

                if (DEBUG_MODE) {
                    cv::imshow("Debug: Intersection Copy", debug_vis);
                    cv::waitKey(1);
                }

                // 최종 결과 반환
                return clean_result;
            });
            break;

        case DistortionType::CLEAN:
        default:
            // 전략: 표준 CLAHE + 적응형 이진화
            this->addStep([](const cv::Mat& img) -> cv::Mat {
                cv::Mat enhanced;
                auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
                clahe->apply(img, enhanced);
                return enhanced;
            });
            // ... (기본 이진화 추가)
            break;
    }
}

// 두 직선 (AB, CD)의 교차점을 구하는 함수
cv::Point2f ImageProcessor::getIntersection(cv::Point A, cv::Point B, cv::Point C, cv::Point D) {
    // 직선 방정식: ax + by = c
    double a1 = B.y - A.y;
    double b1 = A.x - B.x;
    double c1 = a1 * A.x + b1 * A.y;

    double a2 = D.y - C.y;
    double b2 = C.x - D.x;
    double c2 = a2 * C.x + b2 * C.y;

    double det = a1 * b2 - a2 * b1;

    if (std::abs(det) < 1e-9) { // 평행선인 경우
        return cv::Point2f(-1, -1); 
    }

    double x = (b2 * c1 - b1 * c2) / det;
    double y = (a1 * c2 - a2 * c1) / det;
    return cv::Point2f((float)x, (float)y);
}