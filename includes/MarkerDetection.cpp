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
    result.ids.clear();
    result.rvecs.clear();
    result.tvecs.clear();
    result.corners.clear();

    std::vector<std::vector<cv::Point2f>> rejectedCandidates;

    // m_dictionary와 m_detectorParams는 Ptr 타입이어야 함
    cv::aruco::detectMarkers(
        inputImage,
        m_dictionary,       // (Ptr<Dictionary> 타입)
        result.corners,
        result.ids,
        m_detectorParams,   // (Ptr<DetectorParameters> 타입)
        rejectedCandidates
    );

    if (result.ids.empty()) {
        return false;
    }

    cv::aruco::estimatePoseSingleMarkers(
        result.corners,
        m_markerLength,
        m_cameraMatrix,
        m_distCoeffs,
        result.rvecs,
        result.tvecs
    );

    return true;
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