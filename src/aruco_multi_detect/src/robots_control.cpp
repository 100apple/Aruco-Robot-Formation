#include <ros/ros.h>
#include <aruco_multi_detect/ArUcoMarkers.h>
#include <aruco_multi_detect/VirtualLeader.h>
#include <aruco_multi_detect/NashEquilibrium.h>
#include <geometry_msgs/Twist.h>
#include <aruco_multi_detect/TwistStamped.h> // [新增] 引入新的消息头文件
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <map>
#include <signal.h>

using namespace Eigen;
using namespace std;

// 用于存储每个机器人的完整状态 [x, y, theta, v, omega]
struct RobotFullState {
    VectorXd state = VectorXd::Zero(5);
    ros::Time timestamp;
};


class AdvancedFormationController {
private:
    ros::NodeHandle nh_;
    ros::Subscriber aruco_sub_;
    vector<ros::Publisher> cmd_vel_pubs_;
    vector<ros::Publisher> cmd_vel_stamped_pubs_; // [新增] 用于发布带时间戳的速度指令，供数据库节点使用
    ros::Publisher virtual_leader_pub_;
    ros::Publisher nash_pub_;

    // --- 控制与模型参数 ---
    const int N = 4;
    const double l_i_ = 0.1;  // 机器人中心到头部的距离
    const double k2 = 1.0;
    const double k3 = 2.0;
    const double p = 3.0;
    const double gamma_ = 2.0 * (p - 1.0) / p;
    const double boundary_layer_epsilon_ = 0.1; // 平滑边界层厚度

    // --- 编队参数 ---
    MatrixXd A_;          // 邻接矩阵
    MatrixXd Q_kron_;     // 克罗内克积形式的Q矩阵
    VectorXd desired_offsets_d_; // 期望的队形偏移向量

    // --- 状态存储 ---
    map<int, RobotFullState> robot_states_; // id -> [x, y, theta, v, w] 和时间戳
    VectorXd nu_desired_k_minus_1_; // 上一时刻的期望头部速度，用于计算导数
    
    ros::Time start_time_;
    ros::Time last_callback_time_; // 上次回调时间，用于计算dt
    bool first_callback_;

    // --- 虚拟领导者轨迹参数 ---
    const double traj_radius_ = 0.5;
    const double traj_omega_ = 2 * M_PI / 120.0;
    
    // --- 速度限制 ---
    const double MAX_LINEAR_VEL = 0.2;
    const double MAX_ANGULAR_VEL = 0.8;

public:
    AdvancedFormationController() : first_callback_(true) {
        
        // 初始化编队和控制参数
        initializeFormationParameters();
        
        // [修改] 创建控制命令发布器和数据记录发布器
        for(int i = 1; i <= N; i++) {
            string robot_name_prefix = "/bot" + to_string(i);
            
            // 原始的cmd_vel话题，用于实际控制机器人
            string topic_name = robot_name_prefix + "/cmd_vel";
            cmd_vel_pubs_.push_back(nh_.advertise<geometry_msgs::Twist>(topic_name, 10));

            // [新增] 新的cmd_vel_stamped话题，用于数据记录
            string stamped_topic_name = robot_name_prefix + "/cmd_vel_stamped";
            cmd_vel_stamped_pubs_.push_back(nh_.advertise<aruco_multi_detect::TwistStamped>(stamped_topic_name, 10));
        }
        
        // 创建诊断话题发布器
        virtual_leader_pub_ = nh_.advertise<aruco_multi_detect::VirtualLeader>("virtual_leader", 10);
        nash_pub_ = nh_.advertise<aruco_multi_detect::NashEquilibrium>("nash_equilibrium", 10);

        // 订阅ArUco标记
        aruco_sub_ = nh_.subscribe("aruco_markers", 10, &AdvancedFormationController::arucoCallback, this);
        
        nu_desired_k_minus_1_.resize(2 * N);
        nu_desired_k_minus_1_.setZero();
        
        ROS_INFO("Advanced Formation Controller (Realistic Model) initialized.");
        ROS_INFO("Publishing to /botX/cmd_vel and /botX/cmd_vel_stamped.");
        ROS_INFO("Control params: l_i=%.2f, k2=%.2f, k3=%.2f, epsilon=%.3f", l_i_, k2, k3, boundary_layer_epsilon_);
    }

    ~AdvancedFormationController() {
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

    double smoothSign(double value) {
        return std::tanh(value / boundary_layer_epsilon_);
    }

    void initializeFormationParameters() {
        A_.resize(N, N);
        A_ << 0, 1, 0, 1,
              1, 0, 1, 0,
              0, 1, 0, 1,
              1, 0, 1, 0;

        VectorXd in_degree = A_.colwise().sum();
        MatrixXd L = MatrixXd(in_degree.asDiagonal()) - A_;
        VectorXd alpha = VectorXd::Ones(N);
        MatrixXd Q = L + MatrixXd(alpha.asDiagonal());

        MatrixXd P_star(N, 2);
        P_star << -5, -5, 5, -5, -5, 5, 5, 5;
        P_star *= 0.1; // Scaling factor

        MatrixXd D_star(2 * N, N);
        D_star << 0,  10,  0,  10,
                  0,   0, 10,  10,
                 -10,  0, -10,  0,
                  0,   0, 10,  10,
                  0,  10,  0,  10,
                 -10,-10,  0,   0,
                 -10,  0, -10,  0,
                 -10,-10,  0,   0;
        D_star *= 0.1; // Scaling factor

        VectorXd b(2 * N);
        for (int i = 0; i < N; ++i) {
            double sum_dx = 0;
            double sum_dy = 0;
            for (int j = 0; j < N; ++j) {
                sum_dx += A_(i, j) * D_star(2 * i, j);
                sum_dy += A_(i, j) * D_star(2 * i + 1, j);
            }
            b(2 * i)     = sum_dx + P_star(i, 0);
            b(2 * i + 1) = sum_dy + P_star(i, 1);
        }

        MatrixXd I2 = MatrixXd::Identity(2, 2);
        Q_kron_.resize(2 * N, 2 * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Q_kron_.block(2 * i, 2 * j, 2, 2) = Q(i, j) * I2;
            }
        }
        
        desired_offsets_d_ = Q_kron_.colPivHouseholderQr().solve(b);
        ROS_INFO("Formation matrices and desired offsets initialized.");
    }

    void arucoCallback(const aruco_multi_detect::ArUcoMarkers::ConstPtr& msg) {
        ros::Time current_time = ros::Time::now();
        if(first_callback_) {
            start_time_ = current_time;
            last_callback_time_ = current_time;
            first_callback_ = false;
        }

        double dt = (current_time - last_callback_time_).toSec();
        if (dt < 1e-4) {
            return;
        }
        last_callback_time_ = current_time;

        // 步骤 0: 状态估计 (v, w)
        for(const auto& marker : msg->markers) {
            if(marker.id >= 1 && marker.id <= N) {
                RobotFullState current_reading;
                current_reading.state << marker.world_x, marker.world_y, marker.world_z, 0, 0; // world_z 为 yaw
                current_reading.timestamp = current_time;

                if(robot_states_.count(marker.id)) {
                    const RobotFullState& last_state = robot_states_.at(marker.id);
                    double state_dt = (current_reading.timestamp - last_state.timestamp).toSec();
                    if (state_dt > 1e-4) {
                        double dx = current_reading.state(0) - last_state.state(0);
                        double dy = current_reading.state(1) - last_state.state(1);
                        double d_theta = current_reading.state(2) - last_state.state(2);
                        d_theta = atan2(sin(d_theta), cos(d_theta));
                        
                        double estimated_v = sqrt(dx*dx + dy*dy) / state_dt;
                        double estimated_w = d_theta / state_dt;

                        current_reading.state(3) = estimated_v;
                        current_reading.state(4) = estimated_w;
                    }
                }
                robot_states_[marker.id] = current_reading;
            }
        }

        if(robot_states_.size() < N) {
            ROS_WARN_THROTTLE(2.0, "Not all robots detected. Skipping control calculation.");
            return;
        }
        
        // 1. 计算头部状态 [xi_h, nu_h]
        VectorXd xi_head_k(2 * N);
        VectorXd nu_head_k(2 * N);
        for (int i = 0; i < N; ++i) {
            int robot_id = i + 1;
            const auto& state = robot_states_.at(robot_id).state;
            double x = state(0), y = state(1), theta = state(2), v = state(3), w = state(4);
            
            xi_head_k(2 * i)     = x + l_i_ * cos(theta);
            xi_head_k(2 * i + 1) = y + l_i_ * sin(theta);
            nu_head_k(2 * i)     = v * cos(theta) - l_i_ * w * sin(theta);
            nu_head_k(2 * i + 1) = v * sin(theta) + l_i_ * w * cos(theta);
        }

        // 2. 计算虚拟领导者状态和期望的头部位置 (纳什均衡点)
        double elapsed_time = (current_time - start_time_).toSec();
        Vector2d xi0_k;
        xi0_k << 1.5+traj_radius_ * cos(traj_omega_ * elapsed_time),
                 2+traj_radius_ * sin(traj_omega_ * elapsed_time);
        Vector2d xi0_dot;
        xi0_dot << -traj_radius_ * traj_omega_ * sin(traj_omega_ * elapsed_time),
                    traj_radius_ * traj_omega_ * cos(traj_omega_ * elapsed_time);
        
        VectorXd xi0_replicated = xi0_k.replicate(N, 1);
        VectorXd xi_star_k = xi0_replicated + desired_offsets_d_; 

        // 3. 计算二阶编队控制律
        VectorXd Gamma = Q_kron_ * (xi_head_k - xi_star_k);

        VectorXd signed_gamma_term(2 * N);
        for(int i = 0; i < 2 * N; ++i) {
            signed_gamma_term(i) = smoothSign(Gamma(i)) * pow(abs(Gamma(i)), 1.0/p);
        }
        VectorXd nu_desired_k = -k2 * signed_gamma_term + xi0_dot.replicate(N, 1);
        
        VectorXd d_nu_desired = (nu_desired_k - nu_desired_k_minus_1_) / dt;
        VectorXd velocity_error = nu_head_k - nu_desired_k;

        VectorXd signed_error_term(2 * N);
        for(int i=0; i<2*N; ++i) {
            signed_error_term(i) = smoothSign(velocity_error(i)) * pow(abs(velocity_error(i)), gamma_);
        }
        VectorXd u_head = d_nu_desired - k3 * signed_error_term;
        
        nu_desired_k_minus_1_ = nu_desired_k;

        // 4. 模型逆变换 & 计算最终的速度指令
        for (int i = 0; i < N; ++i) {
            int robot_id = i + 1;
            const auto& state = robot_states_.at(robot_id).state;
            double theta = state(2), v = state(3), w = state(4);
            double ux_h = u_head(2 * i);
            double uy_h = u_head(2 * i + 1);

            Matrix2d M;
            M << cos(theta), -l_i_ * sin(theta),
                 sin(theta),  l_i_ * cos(theta);

            Vector2d compensation;
            compensation << ux_h + v * w * sin(theta) + l_i_ * w * w * cos(theta),
                            uy_h - v * w * cos(theta) + l_i_ * w * w * sin(theta);
            
            Vector2d u_actual = M.inverse() * compensation;
            double u_v = u_actual(0);
            double u_w = u_actual(1);

            double v_cmd = v + u_v * dt;
            double w_cmd = w + u_w * dt;

            v_cmd = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, v_cmd));
            w_cmd = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, w_cmd));

            // [修改] 发布两种类型的速度指令
            
            // 1. 发布原始的 Twist 指令，用于机器人控制
            geometry_msgs::Twist cmd;
            cmd.linear.x = v_cmd;
            cmd.angular.z = w_cmd;
            cmd_vel_pubs_[i].publish(cmd);

            // 2. [新增] 发布带时间戳的 TwistStamped 指令，用于数据记录
            aruco_multi_detect::TwistStamped cmd_stamped;
            cmd_stamped.header.stamp = current_time; // 使用本次回调开始时的时间戳
            cmd_stamped.header.frame_id = "bot" + to_string(robot_id); // 可选，但作为良好实践添加
            cmd_stamped.twist = cmd; // 直接将上面创建的cmd消息赋值给内部的twist成员
            cmd_vel_stamped_pubs_[i].publish(cmd_stamped);
        }
        
        // 5. 发布诊断信息
        publishDiagnostics(current_time, xi0_k, xi0_dot, xi_star_k);
    }
    
    void publishDiagnostics(const ros::Time& stamp, const Vector2d& xi0, const Vector2d& xi0_dot, const VectorXd& xi_star) {
        aruco_multi_detect::VirtualLeader leader_msg;
        leader_msg.header.stamp = stamp;
        leader_msg.header.frame_id = "world";
        leader_msg.position.x = xi0.x();
        leader_msg.position.y = xi0.y();
        leader_msg.velocity.x = xi0_dot.x();
        leader_msg.velocity.y = xi0_dot.y();
        virtual_leader_pub_.publish(leader_msg);

        aruco_multi_detect::NashEquilibrium nash_msg;
        nash_msg.header.stamp = stamp;
        nash_msg.header.frame_id = "world";
        for(int i = 0; i < N; ++i) {
            geometry_msgs::Point p;
            p.x = xi_star(2*i);
            p.y = xi_star(2*i+1);
            nash_msg.nash_points.push_back(p);
        }
        nash_pub_.publish(nash_msg);
    }
};

AdvancedFormationController* controller_ptr = nullptr;

void sigintHandler(int sig) {
    if(controller_ptr) {
        controller_ptr->stopAllRobots();
    }
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "robots_control_node", ros::init_options::NoSigintHandler);
    AdvancedFormationController controller;
    controller_ptr = &controller;
    signal(SIGINT, sigintHandler);
    ros::spin();
    return 0;
}
