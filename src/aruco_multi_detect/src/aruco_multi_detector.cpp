#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <aruco_multi_detect/ArUcoMarkers.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <cmath>

class ArucoMultiDetector {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher markers_pub_;
    
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    cv::Ptr<cv::aruco::DetectorParameters> parameters_;
    
    // 相机内参和畸变系数
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    // 相机外参（世界坐标系到相机坐标系的变换）
    cv::Mat R_;  // 旋转矩阵
    cv::Mat tvec_;  // 平移向量
    
    // 平面坐标系参数
    cv::Mat plane_normal_;  // 平面法向量（世界坐标系下）
    double plane_d_;  // 平面方程 ax+by+cz+d=0 中的d
    
    double marker_length_;
    bool show_window_;
    bool use_camera_topic_;
    cv::VideoCapture cap_;

    // 摄像头参数
    int frame_width_;
    int frame_height_;
    int frame_rate_;
    bool use_mjpeg_;

public:
    ArucoMultiDetector() : it_(nh_), show_window_(true), use_camera_topic_(false) {
        // 加载相机参数
        loadCameraParameters();
        
        // 设置平面坐标系（假设Z=0平面）
        plane_normal_ = (cv::Mat_<double>(3,1) << 0, 0, 1); // 法向量指向Z轴正方向
        plane_d_ = 0;  // 平面方程 z=0
        
        // 从ROS参数服务器加载配置
        nh_.param("show_window", show_window_, true);
        nh_.param("use_camera_topic", use_camera_topic_, false);
        nh_.param("frame_width", frame_width_, 1920); // 默认分辨率
        nh_.param("frame_height", frame_height_, 1080);
        nh_.param("frame_rate", frame_rate_, 30);     // 默认帧率
        nh_.param("use_mjpeg", use_mjpeg_, true);     // 默认尝试使用MJPEG

        dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        parameters_ = cv::aruco::DetectorParameters::create();
        parameters_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; 
        
        if (use_camera_topic_) {
            image_sub_ = it_.subscribe("/camera/image_raw", 1, &ArucoMultiDetector::imageCallback, this);
            ROS_INFO("Using camera topic for input");
        } else {
            cap_.open(0, cv::CAP_V4L2);
            if (!cap_.isOpened()) {
                ROS_ERROR("Failed to open camera device!");
                ros::shutdown();
                return;
            }
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);
            cap_.set(cv::CAP_PROP_FPS, frame_rate_);

            if (use_mjpeg_) {
                cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                ROS_INFO("Attempting to set camera format to MJPEG.");
            } else {
                ROS_INFO("Not attempting to set camera format to MJPEG. Using default.");
            }
            
            ROS_INFO("Using direct camera capture with resolution %dx%d @ %d FPS", frame_width_, frame_height_, frame_rate_);
        }
        
        markers_pub_ = nh_.advertise<aruco_multi_detect::ArUcoMarkers>("aruco_markers", 10);
        image_pub_ = it_.advertise("aruco_image", 1);
        
        marker_length_ = 0.15;
        
        if (show_window_) {
            cv::namedWindow("ArUco Detection", cv::WINDOW_NORMAL);
            cv::resizeWindow("ArUco Detection", 1280, 720);
        }
        
        ROS_INFO("ArUco 2D Detector with Extrinsic Calibration Initialized");
    }
    
    ~ArucoMultiDetector() {
        if (cap_.isOpened()) {
            cap_.release();
        }
        if (show_window_) {
            cv::destroyAllWindows();
        }
    }
    
    

    void loadCameraParameters() {
        // 相机内参
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            1054.165162, 0.000000, 673.673821, 
            0.000000, 1059.149190, 308.096635, 
            0.000000, 0.000000, 1.000000);

        // 畸变系数
        dist_coeffs_ = (cv::Mat_<double>(1, 5) << 
            0.217638, -0.260233, -0.020899, 0.007380, 0.000000);

        // 相机外参（世界坐标系到相机坐标系）
        R_ = (cv::Mat_<double>(3, 3) << 
            0.011825261261867, -0.9992146333344097, 0.03781903920605997,
            -0.9988499207485136, -0.01004652687056673, 0.04688179943780741,
            -0.04646503004170337, -0.03832993384053646, -0.9981842601218492);
            
        tvec_ = (cv::Mat_<double>(3, 1) << 
            3.04094576648655,
            2.221728432474176,
            3.242371409718405);
    }
    
    // 将像素坐标转换为相机坐标系下的3D点（在平面上）
    cv::Mat pixelToCamera(const cv::Point2f& pixel) {
        // 1. 去畸变
        std::vector<cv::Point2f> distorted = {pixel};
        std::vector<cv::Point2f> undistorted;
        cv::undistortPoints(distorted, undistorted, camera_matrix_, dist_coeffs_);
        
        // 2. 构造射线方向（相机坐标系下）
        cv::Mat ray = (cv::Mat_<double>(3,1) << 
            undistorted[0].x, 
            undistorted[0].y, 
            1.0);
        ray /= cv::norm(ray); // 归一化射线向量
        
        // 3. 计算射线与平面的交点
        // 世界坐标系下的平面方程: n·X + d = 0
        // 将平面方程转换到相机坐标系下：n_cam · X_cam + d_cam = 0
        // 其中 X_world = R * X_cam + tvec
        // n_world · (R * X_cam + tvec) + d_world = 0
        // (R_inv * n_world) · X_cam + (n_world · tvec + d_world) = 0
        cv::Mat R_inv = R_.t(); // 旋转矩阵的逆等于其转置
        cv::Mat plane_normal_cam = R_ * plane_normal_;
        double plane_d_cam = plane_d_ - plane_normal_cam.dot(tvec_);

        // 计算交点参数 t = -(n_cam·X0 + d_cam)/(n_cam·v)
        // X0是相机原点 (0,0,0)，所以 n_cam·X0 = 0
        double t = -plane_d_cam / plane_normal_cam.dot(ray);
        
        // 交点坐标
        cv::Mat point_cam = t * ray;
        return point_cam;
    }
    
    // 将相机坐标系下的3D点转换为世界坐标系下的3D点
    cv::Mat cameraToWorld(const cv::Mat& point_cam) {
        // 世界坐标 = R_inv * 相机坐标 - R_inv * tvec
        // 或者更直观的：X_world = R_inv * (X_cam - tvec)
        // 但这里我们已经有了R_和tvec_，是世界到相机的变换
        // 所以，X_cam = R_ * X_world + tvec_
        // 那么 X_world = R_inv * (X_cam - tvec_)
        cv::Mat R_inv = R_.t(); // 旋转矩阵的逆
        return R_inv * (point_cam - tvec_);
    }
    
    // 计算标记在世界坐标系XY平面上的朝向角度
    double calculateMarkerAngle(const std::vector<cv::Point2f>& corners) {
        // 1. 获取标记的两个角点（通常是左上角和右上角，取决于ArUco的角点顺序）
        // ArUco角点顺序：0:左上, 1:右上, 2:右下, 3:左下
        cv::Point2f p0 = corners[0];
        cv::Point2f p1 = corners[1];
        
        // 2. 将这两个像素点转换到相机坐标系下的3D点
        cv::Mat p0_cam = pixelToCamera(p0);
        cv::Mat p1_cam = pixelToCamera(p1);
        
        // 3. 计算相机坐标系下的方向向量
        cv::Mat dir_vec_cam = p1_cam - p0_cam;
        
        // 4. 将方向向量转换到世界坐标系
        // 注意：这里只对向量进行旋转，不进行平移
        cv::Mat R_inv = R_.t(); // 旋转矩阵的逆
        cv::Mat dir_vec_world = R_inv * dir_vec_cam;
        
        // 5. 投影到XY平面并计算角度
        // atan2(y, x) 函数返回的是从X轴正方向到点(x,y)的弧度，范围是(-π, π]
        double theta = atan2(dir_vec_world.at<double>(1), dir_vec_world.at<double>(0));
        
        // 将角度调整到[0, 2π)范围（可选，取决于下游应用需求）
        if (theta < 0) {
            theta += 2 * M_PI;
        }
        
        return theta;
    }
    
    // 计算2D姿态（使用外参转换到世界坐标系）
    cv::Mat calculateWorld2DPose(const std::vector<cv::Point2f>& corners) {
        // 1. 计算中心点（像素坐标）
        cv::Point2f center_pixel(0, 0);
        for (const auto& corner : corners) {
            center_pixel.x += corner.x;
            center_pixel.y += corner.y;
        }
        center_pixel.x /= 4;
        center_pixel.y /= 4;
        
        // 2. 将中心点转换到相机坐标系下的3D点（在平面上）
        cv::Mat center_cam = pixelToCamera(center_pixel);
        
        // 3. 转换到世界坐标系
        cv::Mat center_world = cameraToWorld(center_cam);
        
        // 4. 计算角度（基于标记方向）
        double theta = calculateMarkerAngle(corners);
        
        // 返回结果 (X,Y,θ)
        cv::Mat result = (cv::Mat_<double>(3,1) << 
            center_world.at<double>(0), 
            center_world.at<double>(1), 
            theta);
        return result;
    }

    // 在图像上绘制ArUco标记信息
    void draw2DInfo(cv::Mat& image, 
                   const std::vector<cv::Point2f>& corners, 
                   int id, 
                   const cv::Mat& pose) {
        // 绘制标记边界
        cv::aruco::drawDetectedMarkers(image, std::vector<std::vector<cv::Point2f>>{corners}, 
                                      std::vector<int>{id});
        
        // 计算中心点
        cv::Point2f center(0, 0);
        for (const auto& corner : corners) {
            center.x += corner.x;
            center.y += corner.y;
        }
        center.x /= 4;
        center.y /= 4;
        
        // 绘制坐标轴（表示标记的朝向）
        double theta = pose.at<double>(2); // 从姿态中获取角度
        // X轴方向（红色）
        cv::Point2f x_axis_end(center.x + 50 * cos(theta), center.y + 50 * sin(theta));
        // Y轴方向（绿色），与X轴垂直
        cv::Point2f y_axis_end(center.x + 50 * cos(theta + M_PI/2), center.y + 50 * sin(theta + M_PI/2));
        
        cv::line(image, center, x_axis_end, cv::Scalar(0, 0, 255), 2); // 红色X轴
        cv::line(image, center, y_axis_end, cv::Scalar(0, 255, 0), 2); // 绿色Y轴
        
        // 显示坐标和角度
        std::string info = cv::format("ID:%d (X:%.2fm, Y:%.2fm, Theta:%.1f deg)", 
                                    id, 
                                    pose.at<double>(0), 
                                    pose.at<double>(1), 
                                    pose.at<double>(2)*180/M_PI); // 弧度转角度
        cv::putText(image, info, cv::Point(center.x + 20, center.y - 20),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2); // 黄绿色文本
    }
    
    void run() {
        if (use_camera_topic_) {
            ros::spin(); // 如果使用ROS话题，则进入ROS事件循环
        } else {
            // 如果直接从摄像头捕获，则在主循环中读取帧并处理
            ros::Rate rate(frame_rate_); // 尝试以设定的帧率运行
            while (ros::ok()) {
                cv::Mat frame;
                if (cap_.read(frame)) {
                    // 直接处理cv::Mat，避免不必要的cv_bridge转换
                    processFrame(frame); 
                } else {
                    ROS_WARN("Failed to capture frame from camera.");
                }
                ros::spinOnce(); // 处理ROS回调（如果image_pub_有订阅者）
                rate.sleep(); // 控制循环频率
            }
        }
    }
    
    // 当使用ROS话题作为输入时，此回调函数会被调用
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            // 将ROS图像消息转换为OpenCV Mat
            cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;
            processFrame(frame);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("CV bridge error: %s", e.what());
        }
    }
    
    void processFrame(cv::Mat& frame) {
        cv::Mat display_frame = frame.clone(); // 克隆图像用于显示，不影响原始帧
        std::vector<int> marker_ids;
        std::vector<std::vector<cv::Point2f>> marker_corners;
        
        // 检测标记
        cv::aruco::detectMarkers(frame, dictionary_, marker_corners, marker_ids, parameters_);
        
        if (!marker_ids.empty()) {
            aruco_multi_detect::ArUcoMarkers markers_msg;
            markers_msg.header.stamp = ros::Time::now();
            markers_msg.header.frame_id = "world";  // 使用世界坐标系
            
            for (size_t i = 0; i < marker_ids.size(); i++) {
                // 计算中心点（像素坐标）
                cv::Point2f center(0, 0);
                for (const auto& corner : marker_corners[i]) {
                    center.x += corner.x;
                    center.y += corner.y;
                }
                center.x /= 4;
                center.y /= 4;
                
                // 计算2D姿态（使用外参转换到世界坐标系）
                cv::Mat world_pose = calculateWorld2DPose(marker_corners[i]);
                
                // 填充ROS消息
                aruco_multi_detect::ArUcoMarker marker;
                marker.id = marker_ids[i];
                marker.pixel_x = center.x;
                marker.pixel_y = center.y;
                marker.world_x = world_pose.at<double>(0); // X (m)
                marker.world_y = world_pose.at<double>(1); // Y (m)
                marker.world_z = world_pose.at<double>(2); // θ (rad) [0, 2π)
                
                markers_msg.markers.push_back(marker);
                
                // 可视化
                draw2DInfo(display_frame, marker_corners[i], marker_ids[i], world_pose);
            }
            
            markers_pub_.publish(markers_msg); // 发布ArUco标记信息
        }
        
        // 发布带标记的图像（用于可视化）
        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(
            std_msgs::Header(), "bgr8", display_frame).toImageMsg();
        image_pub_.publish(img_msg);
        
        if (show_window_) {
            cv::imshow("ArUco Detection", display_frame);
            cv::waitKey(1); // 必须有waitKey才能显示图像
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "aruco_2d_detector");
    ArucoMultiDetector detector;
    detector.run();
    return 0;
}

