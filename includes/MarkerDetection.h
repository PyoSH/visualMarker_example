//
// Created by Seunghyun Pyo on 2025. 10. 31..
//

#ifndef MARKERDETECTION_H
#define MARKERDETECTION_H

#include<opencv2/opencv.hpp>
#include<opencv2/aruco.hpp>
#include<vector>

struct ArucoResult {
    std::vector<int> ids;                               // 검출된 마커 ID 목록
    std::vector<cv::Vec3d> rvecs;                       // 각 마커의 회전 벡터 (Rodrigues)
    std::vector<cv::Vec3d> tvecs;                       // 각 마커의 변환 벡터 (카메라 좌표계 기준)
    std::vector<std::vector<cv::Point2f>> corners;      // 각 마커의 2D 코너 좌표
};

class MarkerDetection {
public:
    /**
     * @brief 생성자
     * @param cameraMatrix 카메라 내부 파라미터 행렬 (3x3)
     * @param distCoeffs 카메라 왜곡 계수 (1x5 또는 1x4 등)
     * @param markerLengthMeters 마커의 실제 한 변 길이 (미터 단위)
     * @param dictionaryId 사용할 Aruco 딕셔너리 ID (예: cv::aruco::DICT_6X6_250)
     */
    MarkerDetection(cv::Mat cameraMatrix, cv::Mat distCoeffs, float markerLengthMeters,
                       int dictionaryId = cv::aruco::DICT_6X6_250);

    /**
     * @brief 입력 이미지에서 Aruco 마커를 검출하고 Pose를 추정합니다.
     * @param inputImage 입력 이미지 (cv::Mat)
     * @param result 검출 결과를 저장할 ArucoResult 객체 (참조 전달)
     * @return 마커가 1개 이상 검출되었으면 true, 아니면 false
     */
    bool estimatePose(const cv::Mat& inputImage, ArucoResult& result);

    /**
     * @brief 검출 결과(마커 테두리, ID, 좌표축)를 입력 이미지에 그립니다.
     * @param image 결과를 그릴 원본 이미지 (수정됨)
     * @param result estimatePose() 함수에서 반환된 결과 구조체
     */
    void drawResults(cv::Mat& image, const ArucoResult& result);

private:
    cv::Mat m_cameraMatrix;
    cv::Mat m_distCoeffs;
    float m_markerLength;
    cv::Ptr<cv::aruco::Dictionary> m_dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> m_detectorParams;
};

#endif //MARKERDETECTION_H
