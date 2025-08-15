#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <aruco_multi_detect/ArUcoMarkers.h>
#include <aruco_multi_detect/VirtualLeader.h>
#include <aruco_multi_detect/NashEquilibrium.h>
#include <aruco_multi_detect/TwistStamped.h>
#include <geometry_msgs/Twist.h>
#include <mysql_driver.h>
#include <mysql_connection.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>



using namespace message_filters;
using namespace aruco_multi_detect;
using namespace geometry_msgs; // 这里的 Twist 仍然是指 geometry_msgs::Twist

class MysqlWriteNode {
private:
    ros::NodeHandle nh_;
    std::unique_ptr<sql::Connection> con_;
    std::string table_name_;

    // 修改同步策略，使用 TwistStamped 代替 Twist
    typedef sync_policies::ApproximateTime<
        ArUcoMarkers,
        VirtualLeader,
        NashEquilibrium,
        TwistStamped, // <-- 修改
        TwistStamped, // <-- 修改
        TwistStamped, // <-- 修改
        TwistStamped  // <-- 修改
    > MySyncPolicy;
    typedef Synchronizer<MySyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;

    message_filters::Subscriber<ArUcoMarkers> markers_sub_;
    message_filters::Subscriber<VirtualLeader> leader_sub_;
    message_filters::Subscriber<NashEquilibrium> nash_sub_;
    // 修改订阅者类型，使用 TwistStamped
    std::vector<std::shared_ptr<message_filters::Subscriber<TwistStamped>>> cmd_vel_subs_; // <-- 修改

public:
    MysqlWriteNode() {
        // 1. 初始化数据库连接并创建新表
        setupDatabase();

        // 2. 设置话题订阅
        markers_sub_.subscribe(nh_, "aruco_markers", 10);
        leader_sub_.subscribe(nh_, "virtual_leader", 10);
        nash_sub_.subscribe(nh_, "nash_equilibrium", 10);

        for (int i = 1; i <= 4; ++i) {
            auto sub = std::make_shared<message_filters::Subscriber<TwistStamped>>(); // <-- 修改
            // 订阅新的话题名称
            sub->subscribe(nh_, "/bot" + std::to_string(i) + "/cmd_vel_stamped", 10); // <-- 修改话题名称
            cmd_vel_subs_.push_back(sub);
        }

        // 3. 设置时间同步器
        sync_.reset(new Sync(MySyncPolicy(10),
                              markers_sub_,
                              leader_sub_,
                              nash_sub_,
                              *cmd_vel_subs_[0],
                              *cmd_vel_subs_[1],
                              *cmd_vel_subs_[2],
                              *cmd_vel_subs_[3]));
        // 修改回调函数签名，使用 TwistStamped
        sync_->registerCallback(boost::bind(&MysqlWriteNode::callback, this, _1, _2, _3, _4, _5, _6, _7));

        ROS_INFO("MySQL writer node initialized. Writing to table: %s", table_name_.c_str());
    }

    void setupDatabase() {
        try {
            sql::mysql::MySQL_Driver* driver = sql::mysql::get_mysql_driver_instance();
            con_.reset(driver->connect("tcp://127.0.0.1:3306", "car_user", "StrongPassword123!"));
            con_->setSchema("car_tracking");

            // 创建一个基于当前时间戳的唯一表名
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << "experiment_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
            table_name_ = ss.str();

            std::unique_ptr<sql::Statement> stmt(con_->createStatement());
            std::string create_table_sql = "CREATE TABLE " + table_name_ + " ("
                "timestamp DOUBLE,"
                "leader_x DOUBLE, leader_y DOUBLE,"
                "robot_id INT,"
                "actual_x DOUBLE, actual_y DOUBLE,"
                "nash_x DOUBLE, nash_y DOUBLE,"
                "linear_vel DOUBLE, angular_vel DOUBLE,"
                "PRIMARY KEY (timestamp, robot_id)" // <-- 修改这里
                ")";

            stmt->execute(create_table_sql);

        } catch (sql::SQLException &e) {
            ROS_ERROR("MySQL Error: %s (MySQL error code: %d, SQLState: %s)", e.what(), e.getErrorCode(), e.getSQLState().c_str());
            ros::shutdown();
        }
    }

    // 修改回调函数签名，使用 TwistStamped
    void callback(const ArUcoMarkers::ConstPtr& markers_msg,
                  const VirtualLeader::ConstPtr& leader_msg,
                  const NashEquilibrium::ConstPtr& nash_msg,
                  const TwistStamped::ConstPtr& cmd1, // <-- 修改
                  const TwistStamped::ConstPtr& cmd2, // <-- 修改
                  const TwistStamped::ConstPtr& cmd3, // <-- 修改
                  const TwistStamped::ConstPtr& cmd4) // <-- 修改
    {
        try {
            std::unique_ptr<sql::PreparedStatement> pstmt(con_->prepareStatement(
                "INSERT INTO " + table_name_ + " (timestamp, leader_x, leader_y, robot_id, actual_x, actual_y, nash_x, nash_y, linear_vel, angular_vel) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ));

            double timestamp = markers_msg->header.stamp.toSec();
            // 现在需要访问 TwistStamped 内部的 Twist 消息
            const Twist* twists[] = {&cmd1->twist, &cmd2->twist, &cmd3->twist, &cmd4->twist};

            for (const auto& marker : markers_msg->markers) {
                int id = marker.id;
                if (id >= 1 && id <= 4) {
                    pstmt->setDouble(1, timestamp);
                    pstmt->setDouble(2, leader_msg->position.x);
                    pstmt->setDouble(3, leader_msg->position.y);
                    pstmt->setInt(4, id);
                    pstmt->setDouble(5, marker.world_x);
                    pstmt->setDouble(6, marker.world_y);
                    pstmt->setDouble(7, nash_msg->nash_points[id-1].x);
                    pstmt->setDouble(8, nash_msg->nash_points[id-1].y);
                    // 访问 TwistStamped 内部的 Twist 消息
                    pstmt->setDouble(9, twists[id-1]->linear.x);
                    pstmt->setDouble(10, twists[id-1]->angular.z);
                    pstmt->executeUpdate();
                }
            }
        } catch (sql::SQLException &e) {
            ROS_ERROR("Database insert failed: %s", e.what());
        }
    }
};;

int main(int argc, char** argv) {
    ros::init(argc, argv, "mysql_write_node");
    MysqlWriteNode node;
    ros::spin();
    return 0;
}
