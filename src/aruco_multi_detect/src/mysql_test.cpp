
#include <ros/ros.h>
#include <vector>
#include <cmath>
#include <random>

// 使用Eigen库进行矩阵运算，这是ROS中处理线性代数的标准方式
#include <Eigen/Dense>

// 包含所有需要发布的消息类型
#include <aruco_multi_detect/ArUcoMarkers.h>
#include <aruco_multi_detect/VirtualLeader.h>
#include <aruco_multi_detect/NashEquilibrium.h>
#include <aruco_multi_detect/TwistStamped.h>
#include <geometry_msgs/Point.h>

// 为了代码简洁，使用命名空间
using namespace aruco_multi_detect;

/**
 * @class FormationSimulationNode
 * @brief 一个ROS节点，用于模拟多机器人编队控制并发布其状态。
 *
 * 该节点的逻辑直接从提供的Python仿真脚本移植而来。
 * 它计算虚拟领导者、纳什均衡点、机器人实际位置和速度，
 * 并将这些数据发布到相应的ROS话题上。
 */
class FormationSimulationNode {
public:
    FormationSimulationNode() : nh_("~"), gen_(rd_()), dis_(-0.25, 0.25) {
        // 1. 初始化发布者
        leader_pub_ = nh_.advertise<VirtualLeader>("/virtual_leader", 10);
        markers_pub_ = nh_.advertise<ArUcoMarkers>("/aruco_markers", 10);
        nash_pub_ = nh_.advertise<NashEquilibrium>("/nash_equilibrium", 10);

        // 为每个机器人创建速度话题的发布者
        for (int i = 1; i <= N; ++i) {
            std::string topic_name = "/bot" + std::to_string(i) + "/cmd_vel_stamped";
            cmd_vel_pubs_.push_back(nh_.advertise<TwistStamped>(topic_name, 10));
        }

        // 2. 初始化仿真参数和矩阵
        initializeParameters();
        initializeMatrices();

        // 3. 设置仿真初始状态
        setupInitialState();

        // 4. 创建一个定时器，以固定的频率(1/dt)调用仿真循环
        timer_ = nh_.createTimer(ros::Duration(dt), &FormationSimulationNode::simulationStep, this);

        ROS_INFO("Formation simulation node started.");
    }

private:
    // ROS相关
    ros::NodeHandle nh_;
    ros::Publisher leader_pub_, markers_pub_, nash_pub_;
    std::vector<ros::Publisher> cmd_vel_pubs_;
    ros::Timer timer_;

    // 仿真参数 (与Python脚本一致)
    const double dt = 0.01;
    const double k2 = 1.0;
    const double p = 3.0;
    const double k_theta = 5.0;
    const double radius = 0.5;
    const double omega = 2 * M_PI / 120.0;
    const int N = 4;
    double current_time_ = 0.0;

    // 状态变量和矩阵 (使用Eigen库)
    Eigen::MatrixXd A_, Q_, Q_kron_, Q_inv_;
    Eigen::VectorXd d_;
    Eigen::Vector2d xi0_k_;              // 当前参考轨迹点
    Eigen::VectorXd xi_star_k_;          // 当前纳什均衡点
    Eigen::VectorXd xi_actual_k_;        // 当前机器人实际位置
    Eigen::VectorXd theta_actual_k_;     // 当前机器人实际朝向

    // 随机数生成器，用于初始扰动
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<> dis_;


    /**
     * @brief 初始化仿真中使用的常量参数。
     */
    void initializeParameters() {
        // 参数已在类成员定义中初始化
        ROS_INFO("Simulation parameters initialized.");
    }

    /**
     * @brief 初始化所有需要的矩阵和向量，例如邻接矩阵、拉普拉斯矩阵等。
     */
    void initializeMatrices() {
        // 邻接矩阵 A
        A_.resize(N, N);
        A_ << 0, 1, 0, 1,
              1, 0, 1, 0,
              0, 1, 0, 1,
              1, 0, 1, 0;

        // 拉普拉斯矩阵 L
        Eigen::VectorXd in_degree = A_.colwise().sum();
        Eigen::MatrixXd L = Eigen::MatrixXd(in_degree.asDiagonal()) - A_;

        // Q 矩阵 (L + Λ)
        Eigen::VectorXd alpha = Eigen::VectorXd::Ones(N);
        Q_ = L + Eigen::MatrixXd(alpha.asDiagonal());

        // 期望相对位置矩阵 D*
        Eigen::MatrixXd D_star(2 * N, N);
        D_star << 0,  10,  0,  10,
                  0,   0, 10,  10,
                 -10,  0, -10,  0,
                  0,   0, 10,  10,
                  0,  10,  0,  10,
                 -10,-10,  0,   0,
                 -10,  0, -10,  0,
                 -10,-10,  0,   0;
        D_star *= 0.06;

        // 参考点相对位置 P*
        Eigen::MatrixXd P_star(N, 2);
        P_star << -5, -5,
                   5, -5,
                  -5,  5,
                   5,  5;
        P_star *= 0.02;

        // 计算 d 向量
        d_.resize(2 * N);
        for (int i = 0; i < N; ++i) {
            double sum_x = (A_.row(i).array() * D_star.row(2 * i).array()).sum();
            double sum_y = (A_.row(i).array() * D_star.row(2 * i + 1).array()).sum();
            d_(2 * i)     = sum_x + alpha(i) * P_star(i, 0);
            d_(2 * i + 1) = sum_y + alpha(i) * P_star(i, 1);
        }

        // 计算 Q_kron (Q 和 2x2单位阵的克罗内克积) 和其逆矩阵
        Eigen::MatrixXd I2 = Eigen::MatrixXd::Identity(2, 2);
        Q_kron_.resize(2 * N, 2 * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Q_kron_.block(2 * i, 2 * j, 2, 2) = Q_(i, j) * I2;
            }
        }
        Q_inv_ = Q_kron_.inverse();

        ROS_INFO("Matrices initialized.");
    }

    /**
     * @brief 设置机器人的初始位置和姿态。
     */
    void setupInitialState() {
        xi_actual_k_.resize(2 * N);
        theta_actual_k_.resize(N);

        // 计算 t=0 时的参考点和纳什均衡点
        xi0_k_ << radius * cos(0), radius * sin(0);
        Eigen::VectorXd xi0_replicated = xi0_k_.replicate(N, 1);
        xi_star_k_ = Q_inv_ * (d_ + xi0_replicated);

        // 设置初始实际位置，并加入随机扰动
        xi_actual_k_ = xi_star_k_;
        for(int i = 0; i < 2 * N; ++i) {
            xi_actual_k_(i) += dis_(gen_); // Python: np.random.randn(2*N) * 0.5
        }
        
        // 初始朝向角为0
        theta_actual_k_.setZero();

        ROS_INFO("Initial state configured.");
    }

    /**
     * @brief 仿真循环，由ROS定时器周期性调用。
     * @param event 定时器事件信息。
     */
    void simulationStep(const ros::TimerEvent& event) {
        ros::Time now = ros::Time::now();

        // 1. 计算当前时刻的参考轨迹和其导数
        xi0_k_ << radius * cos(omega * current_time_),
                  radius * sin(omega * current_time_);
        Eigen::Vector2d xi0_dot;
        xi0_dot << -radius * omega * sin(omega * current_time_),
                    radius * omega * cos(omega * current_time_);

        // 2. 计算当前时刻的纳什均衡点
        Eigen::VectorXd xi0_replicated = xi0_k_.replicate(N, 1);
        xi_star_k_ = Q_inv_ * (d_ + xi0_replicated);

        // 3. 计算 Gamma
        Eigen::VectorXd Gamma = Q_kron_ * xi_actual_k_ - d_ - xi0_replicated;

        // 4. 计算控制输入 nu (全局坐标系下的期望速度)
        Eigen::VectorXd nu = Eigen::VectorXd::Zero(2 * N);
        for (int i = 0; i < N; ++i) {
            Eigen::Vector2d Gamma_i = Gamma.segment<2>(2 * i);
            Eigen::Vector2d nu_i = -k2 * (Gamma_i.array().cwiseSign() * Gamma_i.array().cwiseAbs().pow(1.0/p)).matrix() + xi0_dot;
            nu.segment<2>(2 * i) = nu_i;
        }

        // 5. 差速小车运动学转换与状态更新
        Eigen::VectorXd v_linear(N);
        Eigen::VectorXd omega_actual(N);

        for (int i = 0; i < N; ++i) {
            double vx_des = nu(2 * i);
            double vy_des = nu(2 * i + 1);
            double current_theta = theta_actual_k_(i);

            // 计算线速度 (投影到当前朝向)
            v_linear(i) = cos(current_theta) * vx_des + sin(current_theta) * vy_des;

            // 计算期望角度和角速度
            double error_theta = 0.0;
            if (sqrt(vx_des*vx_des + vy_des*vy_des) > 1e-3) {
                double theta_des = atan2(vy_des, vx_des);
                error_theta = theta_des - current_theta;
                // 归一化角度误差到[-pi, pi]
                error_theta = atan2(sin(error_theta), cos(error_theta));
            }
            omega_actual(i) = k_theta * error_theta;

            // 更新状态 (欧拉积分)
            theta_actual_k_(i) += omega_actual(i) * dt;
            xi_actual_k_(2 * i) += v_linear(i) * cos(current_theta) * dt;
            xi_actual_k_(2 * i + 1) += v_linear(i) * sin(current_theta) * dt;
        }

        // 6. 填充并发布所有消息
        publishMessages(now, v_linear, omega_actual);

        // 7. 更新仿真时间
        current_time_ += dt;
    }

    /**
     * @brief 将计算出的仿真数据填充到ROS消息中并发布。
     * @param stamp 当前的时间戳。
     * @param v_linear 各个机器人的线速度。
     * @param omega_actual 各个机器人的角速度。
     */
    void publishMessages(const ros::Time& stamp, const Eigen::VectorXd& v_linear, const Eigen::VectorXd& omega_actual) {
        // 发布虚拟领导者
        VirtualLeader leader_msg;
        leader_msg.header.stamp = stamp;
        leader_msg.header.frame_id = "world";
        leader_msg.position.x = xi0_k_(0);
        leader_msg.position.y = xi0_k_(1);
        leader_msg.position.z = 0;
        leader_pub_.publish(leader_msg);

        // 发布ArUco标记 (实际位置)
        ArUcoMarkers markers_msg;
        markers_msg.header.stamp = stamp;
        markers_msg.header.frame_id = "world";
        // FIX: Resize the vector first, then access elements directly.
        // This is more robust and efficient than creating temporary objects with push_back.
        markers_msg.markers.resize(N);
        for (int i = 0; i < N; ++i) {
            markers_msg.markers[i].id = i + 1;
            markers_msg.markers[i].world_x = xi_actual_k_(2 * i);
            markers_msg.markers[i].world_y = xi_actual_k_(2 * i + 1);
        }
        markers_pub_.publish(markers_msg);

        // 发布纳什均衡点
        NashEquilibrium nash_msg;
        nash_msg.header.stamp = stamp;
        nash_msg.header.frame_id = "world";
        nash_msg.nash_points.resize(N); // Also resize here for consistency
        for (int i = 0; i < N; ++i) {
            nash_msg.nash_points[i].x = xi_star_k_(2 * i);
            nash_msg.nash_points[i].y = xi_star_k_(2 * i + 1);
            nash_msg.nash_points[i].z = 0;
        }
        nash_pub_.publish(nash_msg);

        // 发布每个机器人的速度指令
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
    ros::init(argc, argv, "formation_simulation_node");
    FormationSimulationNode sim_node;
    ros::spin();
    return 0;
}


