/*
 * db_test_and_plot.cpp
 *
 * 功能:
 * 1. 连接到MySQL数据库。
 * 2. (可选) 清空 'car_positions' 表中的旧数据。
 * 3. 生成并写入两条模拟的车辆轨迹数据到数据库中。
 * - Marker ID 1: 圆形轨迹
 * - Marker ID 2: 正弦曲线轨迹
 * 4. 从数据库中读取所有轨迹数据。
 * 5. 调用 gnuplot 工具将读取到的轨迹可视化。
 *
 * 编译:
 * 使用提供的 CMakeLists.txt 文件进行编译。
 *
 * 依赖:
 * - MySQL Connector/C++
 * - gnuplot (需要预先在系统中安装，例如: sudo apt-get install gnuplot)
 *
 * 作者: Gemini
 * 日期: 2025-08-06
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <map>
#include <fstream>

// MySQL Connector/C++ Headers
#include <mysql_driver.h>
#include <mysql_connection.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 用于存储从数据库读取的数据点
struct PositionData {
    double x;
    double y;
};

// Gnuplot绘图工具的简单C++封装
class Gnuplot {
private:
    FILE *gnuplotPipe;

public:
    Gnuplot() {
        // "-persist" 选项让绘图窗口在程序结束后不关闭
        gnuplotPipe = popen("gnuplot -persist", "w");
        if (!gnuplotPipe) {
            std::cerr << "错误: Gnuplot 启动失败!" << std::endl;
        }
    }

    ~Gnuplot() {
        if (gnuplotPipe) {
            pclose(gnuplotPipe);
        }
    }

    // 发送指令到 gnuplot
    void command(const std::string &cmd) {
        if (gnuplotPipe) {
            fprintf(gnuplotPipe, "%s\n", cmd.c_str());
            fflush(gnuplotPipe);
        }
    }
};


int main() {
    std::cout << "--- 数据库写入与绘图测试程序 ---" << std::endl;

    // --- 1. 连接数据库 ---
    sql::mysql::MySQL_Driver *driver;
    std::unique_ptr<sql::Connection> con;
    std::unique_ptr<sql::Statement> stmt;
    std::unique_ptr<sql::PreparedStatement> pstmt;
    std::unique_ptr<sql::ResultSet> res;

    try {
        driver = sql::mysql::get_mysql_driver_instance();
        con.reset(driver->connect("tcp://127.0.0.1:3306", "car_user", "StrongPassword123!"));
        con->setSchema("car_tracking");
        stmt.reset(con->createStatement());
        std::cout << "[成功] 已连接到 MySQL 数据库 'car_tracking'." << std::endl;
    } catch (sql::SQLException &e) {
        std::cerr << "[错误] 数据库连接失败: " << e.what() << std::endl;
        std::cerr << "MySQL 错误码: " << e.getErrorCode() << ", SQLState: " << e.getSQLState() << std::endl;
        return 1;
    }

    // --- 2. 清空旧数据并写入模拟数据 ---
    try {
        std::cout << "[信息] 正在清空 'car_positions' 表..." << std::endl;
        stmt->execute("TRUNCATE TABLE car_positions");

        std::cout << "[信息] 正在写入模拟数据..." << std::endl;
        pstmt.reset(con->prepareStatement(
            "INSERT INTO car_positions (marker_id, timestamp, world_x, world_y, world_theta, pixel_x, pixel_y) VALUES (?, ?, ?, ?, ?, ?, ?)"
        ));

        // 模拟数据 1: ID=1, 圆形轨迹
        for (int i = 0; i <= 100; ++i) {
            double angle = 2 * M_PI * i / 100.0;
            double x = 5.0 * cos(angle); // 半径为5的圆
            double y = 5.0 * sin(angle);
            pstmt->setInt(1, 1);
            pstmt->setDouble(2, i * 0.1); // 模拟时间戳
            pstmt->setDouble(3, x);
            pstmt->setDouble(4, y);
            pstmt->setDouble(5, angle); // 模拟角度
            pstmt->setDouble(6, 0); // 像素坐标设为0
            pstmt->setDouble(7, 0);
            pstmt->executeUpdate();
        }

        // 模拟数据 2: ID=2, 正弦曲线轨迹
        for (int i = 0; i <= 100; ++i) {
            double x = i * 0.2; // x从0到20
            double y = 3.0 * sin(x); // 振幅为3的正弦波
            pstmt->setInt(1, 2);
            pstmt->setDouble(2, i * 0.1);
            pstmt->setDouble(3, x);
            pstmt->setDouble(4, y);
            pstmt->setDouble(5, 0); // 角度设为0
            pstmt->setDouble(6, 0);
            pstmt->setDouble(7, 0);
            pstmt->executeUpdate();
        }
        std::cout << "[成功] 模拟数据写入完成。" << std::endl;

    } catch (sql::SQLException &e) {
        std::cerr << "[错误] 数据操作失败: " << e.what() << std::endl;
        return 1;
    }

    // --- 3. 读取数据用于绘图 ---
    std::map<int, std::vector<PositionData>> trajectories;
    try {
        std::cout << "[信息] 正在从数据库读取轨迹数据..." << std::endl;
        res.reset(stmt->executeQuery("SELECT marker_id, world_x, world_y FROM car_positions ORDER BY marker_id, timestamp ASC"));

        while (res->next()) {
            int id = res->getInt("marker_id");
            double x = res->getDouble("world_x");
            double y = res->getDouble("world_y");
            trajectories[id].push_back({x, y});
        }
        std::cout << "[成功] 数据读取完成, 共找到 " << trajectories.size() << " 条轨迹。" << std::endl;

    } catch (sql::SQLException &e) {
        std::cerr << "[错误] 数据读取失败: " << e.what() << std::endl;
        return 1;
    }

    // --- 4. 使用 Gnuplot 绘图 ---
    if (trajectories.empty()) {
        std::cout << "[警告] 没有可供绘图的数据。" << std::endl;
        return 0;
    }

    std::cout << "[信息] 正在启动 Gnuplot 进行可视化..." << std::endl;
    Gnuplot plot;
    plot.command("set title '车辆轨迹可视化 (来自数据库)'");
    plot.command("set xlabel 'World X (m)'");
    plot.command("set ylabel 'World Y (m)'");
    plot.command("set grid");
    plot.command("set key top right");
    plot.command("set dashtype 1 (2,2)"); // 定义虚线样式
    plot.command("set dashtype 2 (4,4)"); // 定义点划线样式
    plot.command("set style line 1 lc rgb 'red' lw 2 pt 7 ps 0.5"); // ID 1 样式
    plot.command("set style line 2 lc rgb 'blue' lw 2 pt 5 ps 0.5"); // ID 2 样式
    plot.command("set style line 3 lc rgb 'green' lw 2 pt 9 ps 0.5"); // 其他ID样式

    std::string plot_cmd = "plot ";
    int plot_index = 0;
    std::vector<std::string> temp_files;

    for (auto const& [id, points] : trajectories) {
        if (points.empty()) continue;

        // 创建临时数据文件
        std::string filename = "temp_data_id_" + std::to_string(id) + ".dat";
        temp_files.push_back(filename);
        std::ofstream temp_file(filename);
        for (const auto& p : points) {
            temp_file << p.x << " " << p.y << "\n";
        }
        temp_file.close();

        if (plot_index > 0) {
            plot_cmd += ", ";
        }
        // 根据ID选择线条样式
        int style_index = (id <= 2) ? id : 3;
        plot_cmd += "'" + filename + "' with linespoints title 'Marker ID " + std::to_string(id) + "' ls " + std::to_string(style_index);
        plot_index++;
    }

    if (plot_index > 0) {
        plot.command(plot_cmd);
        std::cout << "[信息] 绘图指令已发送。请查看 Gnuplot 窗口。" << std::endl;
        std::cout << "按 Enter 键退出并清理临时文件..." << std::endl;
        std::cin.get();
    }

    // 清理临时文件
    for (const auto& filename : temp_files) {
        remove(filename.c_str());
    }
    std::cout << "[完成] 程序已退出。" << std::endl;

    return 0;
}
