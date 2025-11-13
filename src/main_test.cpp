#include <opencv2/opencv.hpp>
#include "../includes/MarkerDetection.h"
#include "../includes/imageProcessor.h"
#include "../includes/PoseEvaluator.h"
#include <iostream>
#include <filesystem>

ImageProcessor processor_;


void processAndLogEvaluation(PoseEvaluator& evaluator,
                             const std::string& filename,
                             int id,
                             const cv::Vec3d& tvec,
                             const cv::Vec3d& rvec,
                             double& sumTransError,
                             double& sumRotError,
                             int& counter)
{
    // 1. 평가 수행
    EvaluationResult evaluated = evaluator.evaluate(filename, id, tvec, rvec);

    // 2. 정답이 있는 경우에만 출력 및 누적
    if (evaluated.hasGt) {
        std::cout << "[" << filename << " | ID:" << id << "] ";

        // 오차 출력 (mm 단위, degree 단위)
        // printf를 사용하여 소수점 포맷팅을 깔끔하게 유지
        printf("Err T: %.2f mm, Err R: %.2f deg\n",
               evaluated.transError * 1000.0, evaluated.rotErrorDeg);

        // 3. 통계 누적 (참조 변수 수정)
        sumTransError += evaluated.transError;
        sumRotError += evaluated.rotErrorDeg;
        counter++;
    } else {
        // (선택 사항) 정답이 없을 때 디버그 메시지
        // std::cout << "[Skip] No GT for " << filename << " ID:" << id << std::endl;
    }
}

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

int main() {
    // !!! 이 값들은 반드시 실제 카메라 보정을 통해 얻은 값을 사용해야 함 !!!
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        1803.10147,     0.0,        980.60876,  // fx, 0, cx
        0.0,            1802.02133, 650.23663,  // 0, fy, cy
        0.0,            0.0,        1.0);

    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.388341, 0.183527, -0.000277, 0.001198, 0.0); // k1, k2, p1, p2, k3

    // 마커의 실제 한 변 길이 (미터 단위)
    float markerLength = 0.168; // 168mm = 0.168m

    int dictionary = cv::aruco::DICT_4X4_50;
    MarkerDetection estimator(cameraMatrix, distCoeffs, markerLength, dictionary);

    std::string folderPath = "../test_imgs";
    std::vector<std::string> imagePaths;
    std::string posePath;

    if (!loadFilesFromFolder(folderPath, imagePaths, posePath)) return -1;

    PoseEvaluator eval(posePath);
    double sumTransError = 0.0;
    double sumRotError = 0.0;
    int counter = 0;

    for (const auto& currImgPath : imagePaths) {
        std::cout << "!!---------Process strt---------!!" << std::endl;
        std::string filename = std::filesystem::path(currImgPath).filename().string();
        cv::Mat img = cv::imread(currImgPath);
        cv::Mat img_prc = img.clone();

        /*
         * 1단계!!!!!!
         */
        // ImageProcessor processor_;

        // 단계 1: 그레이스케일 변환 - 람다 함수 형식
        // processor.addStep([](const cv::Mat& img) -> cv::Mat {
        //     cv::Mat gray;
        //     cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        //     return gray;
        // });

        // // 2. 조명 보정 (CLAHE) - 1번의 결과(gray)가 입력됨
        // processor.addStep([](const cv::Mat& img) -> cv::Mat {
        //     cv::Mat enhanced;
        //     auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        //     clahe->apply(img, enhanced);
        //     return enhanced;
        // });

        // // 3. 노이즈 제거 (Blur) - 2번의 결과(enhanced)가 입력됨
        // processor.addStep([](const cv::Mat& img) -> cv::Mat {
        //     cv::Mat blurred;
        //     cv::GaussianBlur(img, blurred, cv::Size(5, 5), 0);
        //     return blurred;
        // });

        // // (Color -> GRAY)
        setupImageProcessor(false, false, false);
        std::vector<cv::Mat> debug_imgs;
        cv::Mat finalImage = processor_.process(img_prc, debug_imgs);
        cv::imshow("Original Image", img);
        cv::imshow("Final processed Image", finalImage);

        /*
         * 2단계!!!!!!
         */
        ArucoResult results; // 결과를 저장할 구조체

        cv::Mat processImage = finalImage.clone();
        bool found = estimator.estimatePose(processImage, results);

        if (found) {
            std::cout << "--- Found " << results.ids.size() << " Markers ---" << std::endl;
            for (size_t i = 0; i < results.ids.size(); ++i) {
                processAndLogEvaluation(
                    eval,
                    filename,           // 파일명 (확장자 포함)
                    results.ids[i],     // 마커 ID
                    results.tvecs[i],   // 추정 Tvec
                    results.rvecs[i],   // 추정 Rvec
                    sumTransError,    // &참조로 전달 (값이 누적됨)
                    sumRotError,      // &참조로 전달
                    counter          // &참조로 전달
                );
            }
            // estimator.drawResults(img, results);
        } else {
            std::cout << "마커는 찾았으나 Pose 추정에 실패했습니다." << std::endl;
        }

        // cv::imshow("Aruco Pose Estimation", processImage);

        cv::waitKey(0);
    }

    return 0;
}