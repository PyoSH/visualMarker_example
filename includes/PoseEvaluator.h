//
// Created by Seunghyun Pyo on 2025. 11. 12..
//

#ifndef POSE_EVALUATOR_H
#define POSE_EVALUATOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

// 정답(Ground Truth) 데이터 구조체
struct GroundTruth {
    std::string imageFile;
    int markerId;
    cv::Vec3d tvec; // x, y, z
    cv::Vec4d quat; // ox, oy, oz, ow (Quaternion)
};

// 평가 결과 구조체
struct EvaluationResult {
    bool hasGt;             // 정답 데이터 존재 여부
    double transError;      // 이동 오차 (미터)
    double rotErrorDeg;     // 회전 오차 (도, Degree)
    cv::Vec3d gtTvec;       // 정답 Tvec (디버깅용)
};

bool loadFilesFromFolder(const std::string& folderPath,
                         std::vector<std::string>& imagePaths,
                         std::string& posePath);

class PoseEvaluator {
public:
    // 생성자에서 CSV 파일을 로드합니다.
    PoseEvaluator(const std::string& csvPath);

    // 이미지 이름과 마커 ID로 오차를 계산합니다.
    EvaluationResult evaluate(const std::string& imageName, int markerId,
                              const cv::Vec3d& estTvec, const cv::Vec3d& estRvec);

private:
    // 빠른 검색을 위해 Map 사용 (Key: "filename_markerID")
    std::map<std::string, GroundTruth> m_gtMap;

    // Key 생성 헬퍼 함수
    std::string makeKey(const std::string& imageName, int markerId);

    // 쿼터니언(x, y, z, w) -> 회전행렬(3x3) 변환 함수
    cv::Mat quaternionToMatrix(const cv::Vec4d& q);
};

#endif //POSE_EVALUATOR_H