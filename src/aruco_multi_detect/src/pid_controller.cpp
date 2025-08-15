

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <aruco_multi_detect/ArUcoMarkers.h>
#include <cmath>
#include <iostream>
#include <limits>

class DiffDriveController {
private:
    ros::NodeHandle nh_;
    ros::Subscriber aruco_sub_;
    ros::Publisher cmd_vel_pub_;
    
    // 控制参数
    double kp_linear_, ki_linear_, kd_linear_;
    double kp_angular_, ki_angular_, kd_angular_;
    double max_linear_speed_, max_angular_speed_;
    double tolerance_distance_, tolerance_angle_;
    
    // 状态变量
    double target_x_, target_y_;
    double error_integral_linear_, error_integral_angular_;
    double last_error_linear_, last_error_angular_;
    bool has_target_;
    bool is_running_;
    
    // 差速小车参数
    double wheel_separation_;  // 轮间距
    double wheel_radius_;      // 轮子半径
    
public:
    DiffDriveController() : 
        target_x_(0), target_y_(0),
        error_integral_linear_(0), error_integral_angular_(0),
        last_error_linear_(0), last_error_angular_(0),
        has_target_(false), is_running_(false) {
        
        // 加载参数
        nh_.param("pid_linear/kp", kp_linear_, 0.8);
        nh_.param("pid_linear/ki", ki_linear_, 0.02);
        nh_.param("pid_linear/kd", kd_linear_, 0.15);
        
        nh_.param("pid_angular/kp", kp_angular_, 1.5);
        nh_.param("pid_angular/ki", ki_angular_, 0.03);
        nh_.param("pid_angular/kd", kd_angular_, 0.25);
        
        nh_.param("max_linear_speed", max_linear_speed_, 0.1);
        nh_.param("max_angular_speed", max_angular_speed_, 0.2);
        nh_.param("tolerance_distance", tolerance_distance_, 0.10);  // 8cm容差
        nh_.param("tolerance_angle", tolerance_angle_, 0.3);        // ~3度容差
        
        // 差速小车机械参数
        nh_.param("wheel_separation", wheel_separation_, 0.2);  // 轮间距30cm
        nh_.param("wheel_radius", wheel_radius_, 0.04);        // 轮子半径7.5cm
        
        // 订阅和发布
        aruco_sub_ = nh_.subscribe("aruco_markers", 1, &DiffDriveController::arucoCallback, this);
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/bot4/cmd_vel", 10);
        
        ROS_INFO("Differential Drive Controller initialized. Waiting for target...");
        controlLoop();
    }
    
    void arucoCallback(const aruco_multi_detect::ArUcoMarkers::ConstPtr& msg) {
        if (!has_target_) return;
        
        for (const auto& marker : msg->markers) {
            if (marker.id == 4) {  // 小车标记
                controlRobot(marker.world_x, marker.world_y, marker.world_z);
                break;
            }
        }
    }
    
    void controlRobot(double current_x, double current_y, double current_theta) {
        // 计算距离和方向误差
        double dx = target_x_ - current_x;
        double dy = target_y_ - current_y;
        double distance_error = sqrt(dx*dx + dy*dy);
        double target_angle = atan2(dy, dx);
        
        // 角度误差归一化到[-π, π]
        double angle_error = target_angle - current_theta;
        while (angle_error > M_PI) angle_error -= 2*M_PI;
        while (angle_error < -M_PI) angle_error += 2*M_PI;
        
        // 检查是否到达目标
        if (distance_error < tolerance_distance_) {
            if (is_running_) {
                ROS_INFO("Target reached! (%.2f, %.2f)", target_x_, target_y_);
                smoothStop();
                has_target_ = false;
                is_running_ = false;
                waitForNewTarget();
            }
            return;
        }
        
        // 两阶段控制：先转向，再前进
        if (fabs(angle_error) > tolerance_angle_) {
            // 转向阶段 - 只控制角速度
            error_integral_angular_ += angle_error;
            double derivative_angular = angle_error - last_error_angular_;
            double angular_speed = kp_angular_ * angle_error + 
                                 ki_angular_ * error_integral_angular_ + 
                                 kd_angular_ * derivative_angular;
            
            // 限制角速度
            angular_speed = std::min(std::max(angular_speed, -max_angular_speed_), max_angular_speed_);
            
            geometry_msgs::Twist cmd_vel;
            cmd_vel.angular.z = angular_speed;
            cmd_vel_pub_.publish(cmd_vel);
            
            last_error_angular_ = angle_error;
        } else {
            // 前进阶段 - 控制线速度和微调角度
            error_integral_linear_ += distance_error;
            double derivative_linear = distance_error - last_error_linear_;
            double linear_speed = kp_linear_ * distance_error + 
                                ki_linear_ * error_integral_linear_ + 
                                kd_linear_ * derivative_linear;
            
            // 微调角度
            error_integral_angular_ += angle_error;
            double derivative_angular = angle_error - last_error_angular_;
            double angular_speed = kp_angular_ * angle_error + 
                                 ki_angular_ * error_integral_angular_ + 
                                 kd_angular_ * derivative_angular;
            
            // 限制速度
            linear_speed = std::min(std::max(linear_speed, -max_linear_speed_), max_linear_speed_);
            angular_speed = std::min(std::max(angular_speed, -max_angular_speed_/2), max_angular_speed_/2);
            
            geometry_msgs::Twist cmd_vel;
            cmd_vel.linear.x = linear_speed;
            cmd_vel.angular.z = angular_speed;
            cmd_vel_pub_.publish(cmd_vel);
            
            last_error_linear_ = distance_error;
            last_error_angular_ = angle_error;
        }
        
        is_running_ = true;
    }
    
    void smoothStop() {
        ROS_DEBUG("Performing smooth stop");
        geometry_msgs::Twist cmd_vel;
        ros::Rate rate(10);
        
        // 渐进减速
        for (int i = 0; i < 5; ++i) {
            cmd_vel.linear.x *= 0.5;
            cmd_vel.angular.z *= 0.5;
            cmd_vel_pub_.publish(cmd_vel);
            rate.sleep();
        }
        
        // 完全停止
        cmd_vel.linear.x = 0;
        cmd_vel.angular.z = 0;
        cmd_vel_pub_.publish(cmd_vel);
    }
    
    void waitForNewTarget() {
        ROS_INFO("Enter new target coordinates (x y):");
        
        double x, y;
        std::cin >> x >> y;
        
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            ROS_WARN("Invalid input! Please enter numbers only.");
            waitForNewTarget();
            return;
        }
        
        setTarget(x, y);
    }
    
    void setTarget(double x, double y) {
        target_x_ = x;
        target_y_ = y;
        error_integral_linear_ = 0;
        error_integral_angular_ = 0;
        last_error_linear_ = 0;
        last_error_angular_ = 0;
        has_target_ = true;
        
        ROS_INFO("New target set: (%.2f, %.2f)", x, y);
    }
    
    void controlLoop() {
        ros::Rate rate(20);  // 20Hz控制频率
        
        while (ros::ok()) {
            if (!has_target_) {
                waitForNewTarget();
            }
            
            ros::spinOnce();
            rate.sleep();
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "diff_drive_controller");
    DiffDriveController controller;
    return 0;
}
