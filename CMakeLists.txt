cmake_minimum_required(VERSION 3.29)

project(visualMarkerDetection)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

set(OpenCV_DIR /opt/homebrew/lib/cmake/opencv4)
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package(catkin REQUIRED COMPONENTS
#        roscpp
#        sensor_msgs
#        image_transport
#        cv_bridge
#)

add_executable(visualMarkerDetection main_test.cpp
        includes/MarkerDetection.cpp
        includes/MarkerDetection.h
        includes/imageProcessor.cpp
        includes/imageProcessor.h)

target_link_libraries(visualMarkerDetection ${OpenCV_LIBS})

#add_executable(program_ROS main_ros.cpp
#        includes/MarkerDetection.cpp
#        includes/MarkerDetection.h
#        includes/imageProcessor.cpp
#        includes/imageProcessor.h)

#target_link_libraries(program_ROS ${OpenCV_LIBS} ${catkin_LIBRARIES})

