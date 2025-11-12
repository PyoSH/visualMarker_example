cmake_minimum_required(VERSION 3.0.2) # ROS Noetic 권장 버전
project(visualMarkerDetection)

## C++ 17 표준 설정 (람다 함수, filesystem 등 사용 시 필수)
add_compile_options(-std=c++17)

## [중요] ROS 패키지 찾기
#find_package(catkin REQUIRED COMPONENTS
#        roscpp
#        sensor_msgs
#        image_transport
#        cv_bridge
#        std_msgs
#)

## [중요] OpenCV 찾기
set(OpenCV_DIR /opt/homebrew/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)

#catkin_package(
#        #  INCLUDE_DIRS include
#        #  LIBRARIES visualMarkerDetection
#        CATKIN_DEPENDS roscpp sensor_msgs image_transport cv_bridge
#)

## 헤더 파일 경로 설정
include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        includes  # 사용자의 헤더 파일이 있는 폴더
)

add_executable(visualMarkerDetection_test
        src/main_test.cpp
        includes/MarkerDetection.cpp
        includes/imageProcessor.cpp
        includes/PoseEvaluator.cpp
)
target_link_libraries(visualMarkerDetection_test
        ${OpenCV_LIBS}
)

#add_executable(aruco_node
#        src/main_ros.cpp              # ROS 메인 코드 (이전 턴의 aruco_processor_node.cpp 등)
#        includes/MarkerDetection.cpp  # 클래스 구현 파일
#        includes/imageProcessor.cpp   # 클래스 구현 파일
#        # PoseEvaluator는 ROS 노드에서 안 쓰면 빼도 됨
#)
#
### 라이브러리 링크 (OpenCV + ROS Catkin 라이브러리)
#target_link_libraries(aruco_node
#        ${OpenCV_LIBS}
#        ${catkin_LIBRARIES}
#)