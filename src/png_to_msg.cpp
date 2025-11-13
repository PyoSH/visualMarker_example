#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h> 
#include <glob.h> // 파일 목록을 가져오기 위한 헤더

// --- 키보드 입력 감지 함수 (변경 없음) ---
int kbhit(void) {
  struct termios oldt, newt;
  int ch;
  int oldf;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);
  if(ch != EOF) {
    ungetc(ch, stdin);
    return 1;
  }
  return 0;
}

// --- 노이즈 헬퍼 함수들 (변경 없음) ---
void addSaltAndPepperNoise(cv::Mat& img, int num_noise_points = 10000) {
    cv::RNG rng(cv::getTickCount()); 
    for (int i = 0; i < num_noise_points; ++i) {
        int r = rng.uniform(0, img.rows);
        int c = rng.uniform(0, img.cols);
        if (rng.uniform(0, 2) == 0) img.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
        else img.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
    }
}

void addOcclusionNoise(cv::Mat& img) {
    cv::RNG rng(cv::getTickCount());
    int box_width = rng.uniform(img.cols / 8, img.cols / 4); 
    int box_height = rng.uniform(img.rows / 8, img.rows / 4);
    int x = rng.uniform(0, img.cols - box_width);
    int y = rng.uniform(0, img.rows - box_height);
    cv::rectangle(img, cv::Point(x, y), cv::Point(x + box_width, y + box_height), cv::Scalar(0, 0, 0), -1); 
}

void addIlluminationNoise(cv::Mat& img, double alpha, int beta) {
    cv::Mat noisy_img;
    img.convertTo(noisy_img, -1, alpha, beta);
    noisy_img.copyTo(img);
}

void addBlurNoise(cv::Mat& img, int kernel_size = 15) {
    int k = (kernel_size < 1) ? 1 : (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;
    cv::GaussianBlur(img, img, cv::Size(k, k), 0);
}

void applyNoiseToImage(cv::Mat& img, const std::string& noise_type) {
    if (noise_type == "saltpepper") addSaltAndPepperNoise(img, 100000); 
    else if (noise_type == "occlusion") addOcclusionNoise(img);
    else if (noise_type == "bright") addIlluminationNoise(img, 1.0, 100);
    else if (noise_type == "dark") addIlluminationNoise(img, 1.0, -100);
    else if (noise_type == "contrast") addIlluminationNoise(img, 1.5, 0);
    else if (noise_type == "blur") addBlurNoise(img, 100); 
}

// --- 파일 목록 가져오기 헬퍼 함수 ---
std::vector<std::string> getImageFiles(const std::string& folder_path) {
    std::vector<std::string> files;
    std::string pattern = folder_path + "/*.png"; // png 파일만 검색
    
    glob_t glob_result;
    // glob 함수로 파일 패턴 검색
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    
    for(unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    
    // 파일 이름순 정렬 (000 -> 001 -> 002 ...)
    std::sort(files.begin(), files.end());
    
    return files;
}

// --- 메인 함수 ---

int main(int argc, char** argv)
{
  ros::init(argc, argv, "png_publisher");
  ros::NodeHandle nh("~");
  image_transport::ImageTransport it_(nh);

  image_transport::Publisher pub = it_.advertise("/pub_filtered_imgs", 1);

  // 폴더 경로 설정 (끝에 '/'가 없으면 추가해주는 것이 안전하지만 여기선 있다고 가정하거나 추가함)
  std::string folder_path = "/home/kriso/catkin_ws/src/visualMarker_example/test_imgs"; 
  nh.param<std::string>("folder_path", folder_path, folder_path);
  
  // 이미지 파일 리스트 로드
  std::vector<std::string> image_files = getImageFiles(folder_path);
  
  if (image_files.empty()) {
      ROS_ERROR("No .png files found in: %s", folder_path.c_str());
      return 1;
  }

  ROS_INFO("Found %lu images in folder.", image_files.size());
  for(const auto& f : image_files) ROS_INFO(" - %s", f.c_str());

  double rate_hz = 10.0;
  nh.param<double>("rate", rate_hz, 10.0);

  std::string noise_type = "saltpepper";
  nh.param<std::string>("noise_type", noise_type, "none");
  ROS_INFO("Initial noise type: '%s'.", noise_type.c_str());

  // 현재 가리키고 있는 이미지 인덱스
  int current_img_index = 0;
  
  cv::Mat current_display_img;

  // 첫 번째 이미지 로드 및 처리
  cv::Mat img_temp = cv::imread(image_files[current_img_index], cv::IMREAD_COLOR);
  if(img_temp.empty()) {
      ROS_ERROR("Failed to load first image.");
      return 1;
  }
  img_temp.copyTo(current_display_img);
  applyNoiseToImage(current_display_img, noise_type);

  ros::Rate rate(rate_hz);

  std::cout << "\n========================================" << std::endl;
  std::cout << " [Image Iterator Mode] " << std::endl;
  std::cout << " Loaded " << image_files.size() << " images." << std::endl;
  std::cout << " Press 'SPACE BAR' to load NEXT image (Cyclic)." << std::endl;
  std::cout << " Press 'Ctrl + C' to exit." << std::endl;
  std::cout << "========================================\n" << std::endl;

  while (ros::ok()) {
    
    if (kbhit()) {
        int key = getchar(); 

        if (key == ' ') {
            // --- 스페이스바: 다음 이미지 로드 ---
            
            // 인덱스 증가 (마지막이면 0으로 순환)
            current_img_index = (current_img_index + 1) % image_files.size();
            
            // 이미지 파일 로드
            std::string next_file = image_files[current_img_index];
            cv::Mat new_original = cv::imread(next_file, cv::IMREAD_COLOR);
            
            if (!new_original.empty()) {
                // 로드 성공 시 노이즈 적용 후 디스플레이 이미지 업데이트
                new_original.copyTo(current_display_img);
                applyNoiseToImage(current_display_img, noise_type);
                // applyNoiseToImage(current_display_img, "blur");
                
                ROS_INFO(" [Update] Image changed to: %s", next_file.c_str());
            } else {
                ROS_WARN(" [Error] Could not read: %s", next_file.c_str());
            }
        }
        else if (key == 3) { // Ctrl+C
            break;
        }
    }

    // 현재 이미지를 지속적으로 발행
    if(!current_display_img.empty()){
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", current_display_img).toImageMsg();
        msg->header.stamp = ros::Time::now();
        msg->header.frame_id = "camera";
        pub.publish(msg);
    }

    ros::spinOnce();
    rate.sleep(); 
  }

  return 0;
}