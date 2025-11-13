//
// Created by Seunghyun Pyo on 2025. 10. 31..
//

#include "MarkerDetection.h"

MarkerDetection::MarkerDetection(cv::Mat cameraMatrix, cv::Mat distCoeffs, float markerLengthMeters, int dictionaryId)
    : m_cameraMatrix(cameraMatrix.clone()),
      m_distCoeffs(distCoeffs.clone()),
      m_markerLength(markerLengthMeters)
{
    m_dictionary = cv::makePtr<cv::aruco::Dictionary>(
        cv::aruco::getPredefinedDictionary(dictionaryId)
    );

    m_detectorParams = cv::makePtr<cv::aruco::DetectorParameters>();
    m_detectorParams->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
}

bool MarkerDetection::estimatePose(const cv::Mat& inputImage, ArucoResult& result) {
    result.ids.clear(); result.rvecs.clear(); result.tvecs.clear(); result.corners.clear();

    cv::Mat curr_img = inputImage.clone();
    float scale_factor = 1.0f; // 현재 축소 비율 (1, 2, 4...)

    // 피라미드 루프 (최대 3단계: 원본 -> 1/2 -> 1/4)
    for (int level = 0; level < 8; ++level) {
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;

        // 1. [검출 단계] 여기서는 카메라 행렬을 넣지 않거나, 무시합니다.
        // detectMarkers는 2D 사각형을 찾는 역할이 주임무입니다.
        // (Refinement를 위해 매트릭스를 넣기도 하지만, 스케일이 다르면 noArray()를 넣는 게 안전합니다)
        cv::aruco::detectMarkers(curr_img, m_dictionary, corners, ids, m_detectorParams);

        // 2. 마커를 찾았나요?
        if (!ids.empty()) {
            // 3. [좌표 복원] 찾은 2D 좌표를 원본 해상도 기준으로 뻥튀기합니다.
            if (scale_factor > 1.0f) {
                for (auto& marker : corners) {
                    for (auto& point : marker) {
                        point.x *= scale_factor; // x * 2
                        point.y *= scale_factor; // y * 2
                    }
                }
            }

            // 4. [Pose 추정 단계] 
            // 이제 좌표가 '원본 기준'으로 돌아왔으므로, '원본 카메라 행렬'을 사용해도 안전합니다!
            cv::aruco::estimatePoseSingleMarkers(
                corners,           // 원본 크기로 복구된 코너들
                m_markerLength, 
                m_cameraMatrix,    // 원본 카메라 매트릭스 (OK!)
                m_distCoeffs, 
                result.rvecs, 
                result.tvecs
            );

            // 결과 저장 및 반환
            result.ids = ids;
            result.corners = corners;
            return true; // 성공적으로 찾았으므로 종료
        }

        // 못 찾았으면 다음 단계를 위해 이미지를 줄입니다.
        // pyrDown은 가우시안 블러를 포함하므로 노이즈/블러 제거 효과가 있습니다.
        cv::pyrDown(curr_img, curr_img);
        scale_factor *= 2.0f; // 스케일 팩터 2배 증가 (나중에 곱해줄 값)
        
        // 이미지가 너무 작으면 중단
        if (curr_img.cols < 100) break;
    }

    return false; // 모든 단계에서 실패
}

void MarkerDetection::drawResults(cv::Mat& image, const ArucoResult& result) {
    if (result.ids.empty()) {
        return;
    }

    cv::aruco::drawDetectedMarkers(image, result.corners, result.ids);

    float axisLength = m_markerLength * 0.5f;
    for (size_t i = 0; i < result.ids.size(); ++i) {
        cv::drawFrameAxes(
            image,
            m_cameraMatrix,
            m_distCoeffs,
            result.rvecs[i],
            result.tvecs[i],
            axisLength
        );
    }
}