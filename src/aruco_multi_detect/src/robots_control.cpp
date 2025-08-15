#include <ros/ros.h>
#include <aruco_multi_detect/ArUcoMarkers.h>
#include <aruco_multi_detect/VirtualLeader.h>
#include <aruco_multi_detect/NashEquilibrium.h>
#include <geometry_msgs/Twist.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <map>
#include <signal.h>

using namespace Eigen;
using namespace std;

class FiniteTimeController {
private:
    ros::NodeHandle nh_;
    ros::Subscriber aruco_sub_;
    vector<ros::Publisher> cmd_vel_pubs_;
    ros::Publisher virtual_leader_pub_;
    ros::Publisher nash_pub_;
    
    // 控制参数
    double k2_, p_, kp_omega_;
    double formation_scale_;
    
    // 编队参数
    MatrixXd A_;    // 邻接矩阵
    MatrixXd D_;    // 期望相对位置矩阵 (8x4)
    MatrixXd P_;    // 虚拟领导者相对位置 (4x2)
    MatrixXd Q_inv; // (L+I)^-1 * I_2
    
    // 状态存储
    map<int, Vector3d> robot_positions_; // id -> [x, y, theta]
    ros::Time start_time_;
    bool first_callback_;
    
    // 虚拟领导者轨迹参数
    double traj_radius_;
    double traj_omega_;
    
    // 速度限制
    const double MAX_LINEAR_VEL = 0.03;
    const double MAX_ANGULAR_VEL = 0.2;
    
public:
    FiniteTimeController() : 
        k2_(1.0), p_(3.0), kp_omega_(5.0), 
        traj_radius_(0.5), traj_omega_(2*M_PI/120.0),
        first_callback_(true) {
        
        // 初始化编队参数 (菱形编队)
        initializeFormationParameters();
        
        // 创建控制命令发布器
        for(int i = 1; i <= 4; i++) {
            string topic_name = "/bot" + to_string(i) + "/cmd_vel";
            cmd_vel_pubs_.push_back(
                nh_.advertise<geometry_msgs::Twist>(topic_name, 10)
            );
        }
        
        // 创建虚拟领导者和纳什均衡发布器
        virtual_leader_pub_ = nh_.advertise<aruco_multi_detect::VirtualLeader>("virtual_leader", 10);
        nash_pub_ = nh_.advertise<aruco_multi_detect::NashEquilibrium>("nash_equilibrium", 10);

        // 订阅ArUco标记
        aruco_sub_ = nh_.subscribe("aruco_markers", 10, 
                                  &FiniteTimeController::arucoCallback, this);
        
        ROS_INFO("Finite Time Controller initialized");
    }

    ~FiniteTimeController() {
        stopAllRobots();
    }

    void stopAllRobots() {
        geometry_msgs::Twist stop_cmd;
        stop_cmd.linear.x = 0;
        stop_cmd.angular.z = 0;
        for(auto& pub : cmd_vel_pubs_) {
            pub.publish(stop_cmd);
        }
        ROS_INFO("Stopping all robots");
    }

    void initializeFormationParameters() {
        // 邻接矩阵 (菱形拓扑)
        A_ = MatrixXd::Zero(4, 4);
        A_ << 0, 1, 0, 1,
              1, 0, 1, 0,
              0, 1, 0, 1,
              1, 0, 1, 0;
        
        // 期望相对位置矩阵 (8x4, 与Python一致)
        D_ = MatrixXd::Zero(8, 4);
        D_ <<  0, 10,  0, 10,
               0,  0, 10, 10,
             -10,  0,-10,  0,
               0,  0, 10, 10,
               0, 10,  0, 10,
             -10,-10,  0,  0,
             -10,  0,-10,  0,
             -10,-10,  0,  0;
        D_ *= 0.06; // 缩放

        // 虚拟领导者相对位置 (4x2)
        P_ = MatrixXd(4, 2);
        P_ << -5, -5,
               5, -5,
              -5,  5,
               5,  5;
        P_ *= 0.02;

        // 计算 Q_inv
        MatrixXd L = A_.rowwise().sum().asDiagonal();
        L = L - A_;
        MatrixXd Q = L + MatrixXd::Identity(4, 4);
        MatrixXd Q_kron = kroneckerProduct(Q, MatrixXd::Identity(2, 2));
        Q_inv = Q_kron.inverse();
    }

    void getVirtualLeaderState(double t, Vector2d& pos, Vector2d& vel) {
        // 圆形轨迹
        pos << traj_radius_ * cos(traj_omega_ * t),
               traj_radius_ * sin(traj_omega_ * t);
        vel << -traj_radius_ * traj_omega_ * sin(traj_omega_ * t),
                traj_radius_ * traj_omega_ * cos(traj_omega_ * t);
    }

    void arucoCallback(const aruco_multi_detect::ArUcoMarkers::ConstPtr& msg) {
        if(first_callback_) {
            start_time_ = ros::Time::now();
            first_callback_ = false;
        }
        // 更新机器人位置
        for(const auto& marker : msg->markers) {
            if(marker.id >= 1 && marker.id <= 4) {
                robot_positions_[marker.id] = Vector3d(marker.world_x, marker.world_y, marker.world_z);
            }
        }
        if(robot_positions_.size() < 4) {
            ROS_WARN_THROTTLE(2.0, "Not all robots detected. Skipping control calculation.");
            return;
        }
        double elapsed_time = (ros::Time::now() - start_time_).toSec();
        
        // 1. 获取虚拟领导者状态
        Vector2d xi0, xi0_dot;
        getVirtualLeaderState(elapsed_time, xi0, xi0_dot);
        
        // 发布虚拟领导者状态
        aruco_multi_detect::VirtualLeader leader_msg;
        leader_msg.header.stamp = ros::Time::now();
        leader_msg.header.frame_id = "world";
        leader_msg.position.x = xi0.x();
        leader_msg.position.y = xi0.y();
        leader_msg.velocity.x = xi0_dot.x();
        leader_msg.velocity.y = xi0_dot.y();
        virtual_leader_pub_.publish(leader_msg);

        // 2. 计算d向量
        VectorXd d = VectorXd::Zero(8);
        for(int i = 0; i < 4; ++i) {
            double sum_x = 0, sum_y = 0;
            for(int j = 0; j < 4; ++j) {
                sum_x += A_(i, j) * D_(2*i, j);
                sum_y += A_(i, j) * D_(2*i+1, j);
            }
            d(2*i) = sum_x + P_(i, 0);
            d(2*i+1) = sum_y + P_(i, 1);
        }

        // 3. 计算纳什均衡点
        VectorXd xi0_tiled = xi0.replicate(4, 1);
        VectorXd xi_star = Q_inv * (d + xi0_tiled);

        // 发布纳什均衡点
        aruco_multi_detect::NashEquilibrium nash_msg;
        nash_msg.header.stamp = ros::Time::now();
        nash_msg.header.frame_id = "world";
        for(int i=0; i<4; ++i) {
            geometry_msgs::Point p;
            p.x = xi_star(2*i);
            p.y = xi_star(2*i+1);
            nash_msg.nash_points.push_back(p);
        }
        nash_pub_.publish(nash_msg);

        // 4. 计算控制并发布
        for(int robot_id = 1; robot_id <= 4; robot_id++) {
            Vector2d xi_i = robot_positions_[robot_id].head<2>();
            Vector2d Gamma_i = Vector2d::Zero();
            for(int j = 1; j <= 4; j++) {
                if(A_(robot_id-1, j-1) > 0) {
                    Vector2d d_ij(D_(2*(robot_id-1), j-1), D_(2*(robot_id-1)+1, j-1));
                    Vector2d xi_j = robot_positions_[j].head<2>();
                    Gamma_i += A_(robot_id-1, j-1) * ((xi_i - xi_j) - d_ij);
                }
            }
            Vector2d p_i = P_.row(robot_id-1);
            Gamma_i += (xi_i - xi0 - p_i);

            // 控制律
            Vector2d nu_i;
            for(int k=0; k<2; ++k) {
                double x = Gamma_i[k];
                nu_i[k] = -k2_ * copysign(pow(fabs(x), 1.0/p_), x) + xi0_dot[k];
            }

            // 差速小车控制
            geometry_msgs::Twist cmd;
            convertToRobotCommand(robot_id, nu_i, cmd);
            cmd_vel_pubs_[robot_id-1].publish(cmd);
        }
    }

    void convertToRobotCommand(int robot_id, const Vector2d& global_vel, geometry_msgs::Twist& cmd) {
        // 获取当前朝向
        Vector3d state = robot_positions_[robot_id];
        double theta = state.z(); // world_z为yaw
        // 期望速度在当前朝向上的投影
        double v = cos(theta) * global_vel.x() + sin(theta) * global_vel.y();
        // 期望速度方向
        double phi = atan2(global_vel.y(), global_vel.x());
        double error_theta = phi - theta;
        // 归一化到[-pi, pi]
        error_theta = atan2(sin(error_theta), cos(error_theta));
        // 角速度
        double omega = kp_omega_ * error_theta;
        // 限幅
        if(v > MAX_LINEAR_VEL) v = MAX_LINEAR_VEL;
        if(v < -MAX_LINEAR_VEL) v = -MAX_LINEAR_VEL;
        if(omega > MAX_ANGULAR_VEL) omega = MAX_ANGULAR_VEL;
        if(omega < -MAX_ANGULAR_VEL) omega = -MAX_ANGULAR_VEL;
        cmd.linear.x = v;
        cmd.angular.z = omega;
    }

    MatrixXd kroneckerProduct(const MatrixXd& A, const MatrixXd& B) {
        MatrixXd C(A.rows() * B.rows(), A.cols() * B.cols());
        for (int i = 0; i < A.rows(); i++) {
            for (int j = 0; j < A.cols(); j++) {
                C.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
            }
        }
        return C;
    }
};

FiniteTimeController* controller_ptr = nullptr;

void sigintHandler(int sig) {
    if(controller_ptr) {
        controller_ptr->stopAllRobots();
    }
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "robots_control_node", ros::init_options::NoSigintHandler);
    FiniteTimeController controller;
    controller_ptr = &controller;
    signal(SIGINT, sigintHandler);
    ros::spin();
    return 0;
}
