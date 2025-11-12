#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <string> // std::string 사용

// --- 노이즈 추가를 위한 헬퍼 함수들 ---

/**
 * @brief Salt & Pepper 노이즈 추가 (흑백 점)
 * @param img (Input/Output) cv::Mat (CV_8UC3, bgr8)
 * @param num_noise_points 노이즈 점 개수
 */
void addSaltAndPepperNoise(cv::Mat& img, int num_noise_points = 10000) {
    // 매번 다른 노이즈를 위해 tick count로 시드 초기화
    cv::RNG rng(cv::getTickCount()); 
    for (int i = 0; i < num_noise_points; ++i) {
        int r = rng.uniform(0, img.rows);
        int c = rng.uniform(0, img.cols);
        
        if (rng.uniform(0, 2) == 0) {
            img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255); // Salt (흰색)
        } else {
            img.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0); // Pepper (검은색)
        }
    }
}

/**
 * @brief Occlusion 노이즈 추가 (검은색 사각형)
 * @param img (Input/Output) cv::Mat (CV_8UC3, bgr8)
 */
void addOcclusionNoise(cv::Mat& img) {
    cv::RNG rng(cv::getTickCount());
    // 이미지 크기의 1/8 ~ 1/4 사이의 랜덤 크기
    int box_width = rng.uniform(img.cols / 8, img.cols / 4); 
    int box_height = rng.uniform(img.rows / 8, img.rows / 4);
    // 랜덤 위치
    int x = rng.uniform(0, img.cols - box_width);
    int y = rng.uniform(0, img.rows - box_height);
    
    // 매 프레임 다른 위치에 검은색 사각형
    cv::rectangle(img, cv::Point(x, y), cv::Point(x + box_width, y + box_height), cv::Scalar(0, 0, 0), -1); 
}

/**
 * @brief 조도 변화 (밝기/대비 조절)
 * @param img (Input/Output) cv::Mat (CV_8UC3, bgr8)
 * @param alpha 대비 (1.0 = 유지)
 * @param beta 밝기 (0 = 유지)
 */
void addIlluminationNoise(cv::Mat& img, double alpha, int beta) {
    // convertTo는 원본을 덮어쓰지 않도록 새 Mat으로 결과를 받습니다.
    cv::Mat noisy_img;
    img.convertTo(noisy_img, -1, alpha, beta);
    noisy_img.copyTo(img); // 결과를 다시 img에 복사
}
/*
* @brief Gaussian Blur 노이즈 추가
 * @param img (Input/Output) cv::Mat (CV_8UC3, bgr8)
 * @param kernel_size 커널 크기 (양의 홀수여야 함)
*/
void addBlurNoise(cv::Mat& img, int kernel_size = 15) {
    // kernel_size는 양의 홀수여야 함
    int k = (kernel_size < 1) ? 1 : (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
    // cv::GaussianBlur(원본, 결과, 커널크기, 시그마)
    cv::GaussianBlur(img, img, cv::Size(k, k), 0);
}

// --- 헬퍼 함수 끝 ---


int main(int argc, char** argv)
{
  ros::init(argc, argv, "png_publisher");
  ros::NodeHandle nh("~"); //  private 노드 핸들 '~' 유지
  image_transport::ImageTransport it_(nh);

  // 토픽명 /aruco_marker_noise_img
  image_transport::Publisher pub = it_.advertise("/aruco_marker_noise_img_PSH", 1);

  // 파라미터 or 실행 시 인자로 PNG 경로 받기
  std::string file_path = "/home/kriso/catkin_ws/src/visualMarker_example/test_imgs/aruco_image_000.png"; // 기본값
  // ~file_path 파라미터로 위 기본값을 덮어쓸 수 있도록 수정
  nh.param<std::string>("file_path", file_path, file_path);
  ROS_INFO("Loading image from: %s", file_path.c_str());

  double rate_hz = 1.0;
  nh.param<double>("rate", rate_hz, 1.0); // Rate도 파라미터로 조절 가능

  // 노이즈 타입 선택을 위한 파라미터
  std::string noise_type = "saltpepper";
  nh.param<std::string>("noise_type", noise_type, "none");
  ROS_INFO("Selected noise type: '%s'. (Options: none, saltpepper, occlusion, bright, dark, contrast, blur)", noise_type.c_str());


  cv::Mat img_original = cv::imread(file_path, cv::IMREAD_COLOR);
  if (img_original.empty()) {
    ROS_ERROR("Failed to load image: %s", file_path.c_str());
    return 1;
  }

  ros::Rate rate(rate_hz);

  while (ros::ok()) {

    // 원본 이미지를 매번 복사 (원본 훼손 방지)
    cv::Mat noisy_img;
    img_original.copyTo(noisy_img);

    // 선택된 노이즈 타입 적용
    if (noise_type == "saltpepper") {
        addSaltAndPepperNoise(noisy_img, 100000); // 10000개의 점
    } 
    else if (noise_type == "occlusion") {
        addOcclusionNoise(noisy_img); // 랜덤 위치에 검은 사각형
    }
    else if (noise_type == "bright") { // 밝게
        addIlluminationNoise(noisy_img, 1.0, 100); // alpha=1.0 (대비 유지), beta=80 (밝게)
    }
    else if (noise_type == "dark") { // 어둡게
        addIlluminationNoise(noisy_img, 1.0, -100); // alpha=1.0 (대비 유지), beta=-80 (어둡게)
    }
    else if (noise_type == "contrast") { // 대비 강하게
        addIlluminationNoise(noisy_img, 1.5, 0); // alpha=1.5 (대비 증가), beta=0 (밝기 유지)
    }
    else if (noise_type == "blur") { //블러링
        addBlurNoise(noisy_img, 200); 
    }// 15x15 커널 크기
    // "none" 이거나 일치하는게 없으면 noisy_img (원본 복사본)이 그대로 사용됨


    //노이즈가 적용된 noisy_img로 sensor_msgs::Image 변환
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", noisy_img).toImageMsg();
    msg->header.stamp = ros::Time::now();
    msg->header.frame_id = "camera";

    pub.publish(msg);

    ROS_INFO_THROTTLE(2.0, "Publishing /aruco_marker_noise_img (Noise: %s)", noise_type.c_str());
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}