#include <ros/ros.h>
#include <vector>
#include <cmath>
#include <random>

#include <Eigen/Dense>

#include <aruco_multi_detect/ArUcoMarkers.h>
#include <aruco_multi_detect/VirtualLeader.h>
#include <aruco_multi_detect/NashEquilibrium.h>
#include <aruco_multi_detect/TwistStamped.h>
#include <geometry_msgs/Point.h>

using namespace aruco_multi_detect;

/**
 * @class FormationSimulationNode
 * @brief 一个ROS节点，用于模拟多机器人编队控制并发布其状态。
 *
 * 该节点的控制算法基于二阶系统和模型逆变换，以匹配实际小车动力学模型。
 * [MODIFIED] 采用了边界层法 (Boundary Layer Method) 来平滑控制输出，消除抖振。
 */
class FormationSimulationNode {
public:
    FormationSimulationNode() : nh_("~"), gen_(rd_()), pos_dis_(-0.75, 0.75), vel_dis_(-0.5, 0.5), ang_dis_(0, 2*M_PI) {
        // 1. 初始化发布者
        leader_pub_ = nh_.advertise<VirtualLeader>("/virtual_leader", 10);
        markers_pub_ = nh_.advertise<ArUcoMarkers>("/aruco_markers", 10);
        nash_pub_ = nh_.advertise<NashEquilibrium>("/nash_equilibrium", 10);

        for (int i = 1; i <= N; ++i) {
            std::string topic_name = "/bot" + std::to_string(i) + "/cmd_vel_stamped";
            cmd_vel_pubs_.push_back(nh_.advertise<TwistStamped>(topic_name, 10));
        }

        // 2. 初始化仿真参数和矩阵
        initializeParameters();
        initializeMatrices();

        // 3. 设置仿真初始状态
        setupInitialState();

        // 4. 创建定时器
        timer_ = nh_.createTimer(ros::Duration(dt), &FormationSimulationNode::simulationStep, this);

        ROS_INFO("Formation simulation node (Realistic Model with Smoothing) started.");
    }

private:
    // ROS
    ros::NodeHandle nh_;
    ros::Publisher leader_pub_, markers_pub_, nash_pub_;
    std::vector<ros::Publisher> cmd_vel_pubs_;
    ros::Timer timer_;

    // 仿真参数
    const double dt = 0.01;
    const int N = 4;
    const double radius = 0.5;
    const double omega = 2 * M_PI / 120.0;
    double current_time_ = 0.0;

    // 控制与模型参数
    const double l_i_ = 0.1; 
    const double k2 = 1.0;
    const double k3 = 2.0;
    const double p = 3.0;
    double gamma_;

    // [NEW] 边界层参数，用于平滑sign()函数，消除抖振
    const double boundary_layer_epsilon_ = 0.1; 

    // 矩阵
    Eigen::MatrixXd A_, Q_kron_;
    Eigen::VectorXd desired_offsets_d_; 

    // 状态变量
    Eigen::MatrixXd actual_states_; // size 5xN: [x, y, theta, v, omega]
    Eigen::VectorXd xi_head_k_;
    Eigen::VectorXd nu_head_k_;
    Eigen::VectorXd nu_desired_k_minus_1_;

    // 辅助变量
    Eigen::Vector2d xi0_k_;
    Eigen::VectorXd xi_star_k_;

    // 随机数生成器
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<> pos_dis_;
    std::uniform_real_distribution<> vel_dis_;
    std::uniform_real_distribution<> ang_dis_;

    // [NEW] 辅助函数，使用tanh实现平滑的sign函数
    double smoothSign(double value) {
        return std::tanh(value / boundary_layer_epsilon_);
    }

    void initializeParameters() {
        gamma_ = 2.0 * (p - 1.0) / p;
        ROS_INFO("Simulation parameters initialized (k2=%.2f, k3=%.2f, l_i=%.2f).", k2, k3, l_i_);
        ROS_INFO("Boundary layer epsilon for smoothing set to: %.3f", boundary_layer_epsilon_);
    }

    void initializeMatrices() {
        A_.resize(N, N);
        A_ << 0, 1, 0, 1,
              1, 0, 1, 0,
              0, 1, 0, 1,
              1, 0, 1, 0;

        Eigen::VectorXd in_degree = A_.colwise().sum();
        Eigen::MatrixXd L = Eigen::MatrixXd(in_degree.asDiagonal()) - A_;
        Eigen::VectorXd alpha = Eigen::VectorXd::Ones(N);
        Eigen::MatrixXd Q = L + Eigen::MatrixXd(alpha.asDiagonal());

        Eigen::MatrixXd P_star(N, 2);
        P_star << -5, -5, 5, -5, -5, 5, 5, 5;
        P_star *= 0.02; // Scaling factor

        Eigen::MatrixXd D_star(2 * N, N);
        D_star << 0,  10,  0,  10,
                  0,   0, 10,  10,
                 -10,  0, -10,  0,
                  0,   0, 10,  10,
                  0,  10,  0,  10,
                 -10,-10,  0,   0,
                 -10,  0, -10,  0,
                 -10,-10,  0,   0;
        D_star *= 0.06; // Scaling factor

        Eigen::VectorXd b(2 * N);
        for (int i = 0; i < N; ++i) {
            double sum_dx = (A_.row(i).array() * D_star.row(2 * i).array()).sum();
            double sum_dy = (A_.row(i).array() * D_star.row(2 * i + 1).array()).sum();
            b(2 * i)     = sum_dx + P_star(i, 0);
            b(2 * i + 1) = sum_dy + P_star(i, 1);
        }

        Eigen::MatrixXd I2 = Eigen::MatrixXd::Identity(2, 2);
        Q_kron_.resize(2 * N, 2 * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Q_kron_.block(2 * i, 2 * j, 2, 2) = Q(i, j) * I2;
            }
        }
        
        desired_offsets_d_ = Q_kron_.colPivHouseholderQr().solve(b);

        ROS_INFO("Matrices and desired formation offsets initialized.");
    }
    
    void setupInitialState() {
        actual_states_.resize(5, N);
        xi_head_k_.resize(2 * N);
        nu_head_k_.resize(2 * N);
        nu_desired_k_minus_1_.resize(2 * N);
        nu_desired_k_minus_1_.setZero();
        xi_star_k_.resize(2 * N);

        xi0_k_ << radius * cos(0), radius * sin(0);

        for (int i = 0; i < N; ++i) {
            double x0 = xi0_k_(0) + pos_dis_(gen_);
            double y0 = xi0_k_(1) + pos_dis_(gen_);
            double theta0 = ang_dis_(gen_);
            double v0 = 1.0 + vel_dis_(gen_);
            double omega0 = 0.1 * vel_dis_(gen_);
            actual_states_.col(i) << x0, y0, theta0, v0, omega0;
        }

        ROS_INFO("Initial state configured randomly.");
    }

    void simulationStep(const ros::TimerEvent& event) {
        ros::Time now = ros::Time::now();

        // 1. 计算头部状态 [xi_h, nu_h]
        for (int i = 0; i < N; ++i) {
            double x = actual_states_(0, i);
            double y = actual_states_(1, i);
            double theta = actual_states_(2, i);
            double v = actual_states_(3, i);
            double w = actual_states_(4, i);

            xi_head_k_(2 * i)     = x + l_i_ * cos(theta);
            xi_head_k_(2 * i + 1) = y + l_i_ * sin(theta);
            nu_head_k_(2 * i)     = v * cos(theta) - l_i_ * w * sin(theta);
            nu_head_k_(2 * i + 1) = v * sin(theta) + l_i_ * w * cos(theta);
        }

        // 2. 计算虚拟领导者状态和期望的头部位置
        xi0_k_ << radius * cos(omega * current_time_), radius * sin(omega * current_time_);
        Eigen::Vector2d xi0_dot;
        xi0_dot << -radius * omega * sin(omega * current_time_), radius * omega * cos(omega * current_time_);
        
        Eigen::VectorXd xi0_replicated = xi0_k_.replicate(N, 1);
        xi_star_k_ = xi0_replicated + desired_offsets_d_; 

        // 3. 计算编队控制律 (作用于头部)
        Eigen::VectorXd Gamma = Q_kron_ * (xi_head_k_ - xi_star_k_);

        // [MODIFIED] 使用平滑的sign函数来计算 nu_desired，避免期望速度突变
        Eigen::VectorXd signed_gamma_term = (
            Gamma.unaryExpr([this](double g){ return smoothSign(g); }).array() * 
            Gamma.array().abs().pow(1.0/p)
        ).matrix();
        Eigen::VectorXd nu_desired_k = -k2 * signed_gamma_term + xi0_dot.replicate(N, 1);
        
        Eigen::VectorXd d_nu_desired = (nu_desired_k - nu_desired_k_minus_1_) / dt;
        Eigen::VectorXd velocity_error = nu_head_k_ - nu_desired_k;

        // [MODIFIED] 使用平滑的sign函数来计算 u_head，消除抖振 (chattering)
        Eigen::VectorXd signed_error_term = (
            velocity_error.unaryExpr([this](double e){ return smoothSign(e); }).array() *
            velocity_error.array().abs().pow(gamma_)
        ).matrix();
        Eigen::VectorXd u_head = d_nu_desired - k3 * signed_error_term;
        
        nu_desired_k_minus_1_ = nu_desired_k;

        // 4. 模型逆变换: u_head -> u_actual
        Eigen::MatrixXd u_actual(2, N);
        for (int i = 0; i < N; ++i) {
            double theta = actual_states_(2, i);
            double v = actual_states_(3, i);
            double w = actual_states_(4, i);
            double ux_h = u_head(2 * i);
            double uy_h = u_head(2 * i + 1);

            Eigen::Matrix2d M;
            M << cos(theta), -l_i_ * sin(theta),
                 sin(theta),  l_i_ * cos(theta);

            Eigen::Vector2d compensation;
            compensation << ux_h + v * w * sin(theta) + l_i_ * w * w * cos(theta),
                            uy_h - v * w * cos(theta) + l_i_ * w * w * sin(theta);
            
            u_actual.col(i) = M.inverse() * compensation;
        }

        // 5. 状态更新
        Eigen::VectorXd v_linear_next(N);
        Eigen::VectorXd omega_angular_next(N);

        for (int i = 0; i < N; ++i) {
            double x = actual_states_(0, i);
            double y = actual_states_(1, i);
            double theta = actual_states_(2, i);
            double v = actual_states_(3, i);
            double w = actual_states_(4, i);
            double u_v = u_actual(0, i);
            double u_w = u_actual(1, i);
            double x_next = x + v * cos(theta) * dt;
            double y_next = y + v * sin(theta) * dt;
            double theta_next = theta + w * dt;
            
            v_linear_next(i) = v + u_v * dt;
            omega_angular_next(i) = w + u_w * dt;

            actual_states_.col(i) << x_next, y_next, theta_next, v_linear_next(i), omega_angular_next(i);
        }

        // 6. 发布消息
        publishMessages(now, v_linear_next, omega_angular_next);

        // 7. 更新仿真时间
        current_time_ += dt;
    }

    void publishMessages(const ros::Time& stamp, const Eigen::VectorXd& v_linear, const Eigen::VectorXd& omega_actual) {
        VirtualLeader leader_msg;
        leader_msg.header.stamp = stamp;
        leader_msg.header.frame_id = "world";
        leader_msg.position.x = xi0_k_(0);
        leader_msg.position.y = xi0_k_(1);
        leader_msg.position.z = 0;
        leader_pub_.publish(leader_msg);

        ArUcoMarkers markers_msg;
        markers_msg.header.stamp = stamp;
        markers_msg.header.frame_id = "world";
        markers_msg.markers.resize(N);
        for (int i = 0; i < N; ++i) {
            markers_msg.markers[i].id = i + 1;
            markers_msg.markers[i].world_x = actual_states_(0, i);
            markers_msg.markers[i].world_y = actual_states_(1, i);
        }
        markers_pub_.publish(markers_msg);

        NashEquilibrium nash_msg;
        nash_msg.header.stamp = stamp;
        nash_msg.header.frame_id = "world";
        nash_msg.nash_points.resize(N);
        for (int i = 0; i < N; ++i) {
            nash_msg.nash_points[i].x = xi_star_k_(2 * i);
            nash_msg.nash_points[i].y = xi_star_k_(2 * i + 1);
            nash_msg.nash_points[i].z = 0;
        }
        nash_pub_.publish(nash_msg);

        for (int i = 0; i < N; ++i) {
            TwistStamped cmd_vel_msg;
            cmd_vel_msg.header.stamp = stamp;
            cmd_vel_msg.header.frame_id = "bot" + std::to_string(i + 1);
            cmd_vel_msg.twist.linear.x = v_linear(i);
            cmd_vel_msg.twist.angular.z = omega_actual(i);
            cmd_vel_pubs_[i].publish(cmd_vel_msg);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "formation_simulation_node_realistic_model");
    FormationSimulationNode sim_node;
    ros::spin();
    return 0;
}
