//
// Created by Seunghyun Pyo on 2025. 11. 12..
//
#include "PoseEvaluator.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

bool loadFilesFromFolder(const std::string& folderPath,
                         std::vector<std::string>& imagePaths,
                         std::string& posePath) {
    // 출력 변수 초기화
    imagePaths.clear();
    posePath = "";

    try {
        // 디렉토리 순회
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                std::string pathStr = entry.path().string();
                std::string ext = entry.path().extension().string();

                // 확장자 소문자 변환 (대소문자 무시를 위해)
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    imagePaths.push_back(pathStr);
                }
                else if (ext == ".csv") {
                    posePath = pathStr;
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "폴더 접근 오류: " << e.what() << std::endl;
        return false;
    }

    // 이미지 경로 정렬
    std::sort(imagePaths.begin(), imagePaths.end());

    std::cout << "총 " << imagePaths.size() << " 장의 이미지 확인" << std::endl;
    if (!posePath.empty()) {
        std::cout << "Pose 데이터 파일 확인: " << posePath << std::endl;
    }

    return true;
}

PoseEvaluator::PoseEvaluator(const std::string& csvPath) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[PoseEvaluator] Error: CSV 파일을 열 수 없습니다: " << csvPath << std::endl;
        return;
    }

    std::string line;
    // 1. 헤더 라인 건너뛰기 (image_file,marker_index,x,y,z,ox,oy,oz,ow)
    std::getline(file, line);

    int loadedCount = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> row;

        while (std::getline(ss, segment, ',')) {
            row.push_back(segment);
        }

        // 데이터 열 개수 확인 (최소 9개)
        if (row.size() < 9) continue;

        GroundTruth gt;
        gt.imageFile = row[0];
        gt.markerId = std::stoi(row[1]);

        // Translation (x, y, z) -> index 2, 3, 4
        gt.tvec = cv::Vec3d(std::stod(row[2]), std::stod(row[3]), std::stod(row[4]));

        // Quaternion (ox, oy, oz, ow) -> index 5, 6, 7, 8
        // 주의: OpenCV 등 수식에서 w, x, y, z 순서인지 x, y, z, w 인지 확인 필요.
        // 여기서는 저장 순서를 ox, oy, oz, ow로 저장하고 변환 함수에서 처리함.
        gt.quat = cv::Vec4d(std::stod(row[5]), std::stod(row[6]), std::stod(row[7]), std::stod(row[8]));

        // 맵에 저장
        m_gtMap[makeKey(gt.imageFile, gt.markerId)] = gt;
        loadedCount++;
    }
    std::cout << "[PoseEvaluator] " << loadedCount << " 개의 정답 데이터를 로드했습니다." << std::endl;
}

std::string PoseEvaluator::makeKey(const std::string& imageName, int markerId) {
    // 파일명과 ID를 조합하여 고유 키 생성
    return imageName + "_" + std::to_string(markerId);
}

cv::Mat PoseEvaluator::quaternionToMatrix(const cv::Vec4d& q) {
    // 입력 q는 (ox, oy, oz, ow) 순서라고 가정합니다.
    double x = q[0];
    double y = q[1];
    double z = q[2];
    double w = q[3];

    // 정규화 (Normalization) - 필수!
    double norm = std::sqrt(x*x + y*y + z*z + w*w);
    if (norm > 0) {
        x /= norm; y /= norm; z /= norm; w /= norm;
    }

    // Quaternion(x, y, z, w) to Rotation Matrix 공식
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        1.0 - 2.0*y*y - 2.0*z*z,       2.0*x*y - 2.0*z*w,       2.0*x*z + 2.0*y*w,
              2.0*x*y + 2.0*z*w, 1.0 - 2.0*x*x - 2.0*z*z,       2.0*y*z - 2.0*x*w,
              2.0*x*z - 2.0*y*w,       2.0*y*z + 2.0*x*w, 1.0 - 2.0*x*x - 2.0*y*y
    );
    return R;
}

EvaluationResult PoseEvaluator::evaluate(const std::string& imageName, int markerId,
                                         const cv::Vec3d& estTvec, const cv::Vec3d& estRvec) {
    EvaluationResult result;
    std::string key = makeKey(imageName, markerId);

    // 1. 정답 데이터 찾기
    if (m_gtMap.find(key) == m_gtMap.end()) {
        result.hasGt = false;
        result.transError = -1.0;
        result.rotErrorDeg = -1.0;
        return result;
    }

    GroundTruth gt = m_gtMap[key];
    result.hasGt = true;
    result.gtTvec = gt.tvec;

    // 2. Translation Error (Euclidean distance)
    // 단위가 같은지(meter vs meter) 확인이 중요함
    result.transError = cv::norm(gt.tvec - estTvec);
    std::cout << "GT position : " << gt.tvec << std::endl;
    std::cout << "EST position: " << estTvec << std::endl;

    // 3. Rotation Error
    // 3-1. GT Quaternion -> Rotation Matrix
    cv::Mat R_gt = quaternionToMatrix(gt.quat);

    // 3-2. Estimated Rodrigues -> Rotation Matrix
    cv::Mat R_est;
    cv::Rodrigues(estRvec, R_est);

    // 3-3. 회전 차이 계산: R_diff = R_gt * R_est^T
    // 두 회전 사이의 "사이각"을 구하는 과정
    cv::Mat R_diff = R_gt * R_est.t();

    // 3-4. 로드리게스 벡터로 변환하여 각도 추출
    cv::Vec3d r_diff_vec;
    cv::Rodrigues(R_diff, r_diff_vec);

    // 벡터의 크기(norm)가 곧 회전축 기준 회전량(radian)입니다.
    double errorRad = cv::norm(r_diff_vec);

    // Radian -> Degree
    result.rotErrorDeg = errorRad * (180.0 / CV_PI);

    return result;
}