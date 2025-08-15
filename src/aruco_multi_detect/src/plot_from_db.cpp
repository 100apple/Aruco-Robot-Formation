#include <mysql_driver.h>
#include <mysql_connection.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Point {
    double x;
    double y;
    double theta;
    double t;
};

class GnuplotPipe {
public:
    GnuplotPipe() {
        pipe = popen("gnuplot -persist", "w");
        if (!pipe) throw std::runtime_error("Cannot open gnuplot");
    }
    ~GnuplotPipe() {
        if (pipe) pclose(pipe);
    }
    void send(const std::string& cmd) {
        fprintf(pipe, "%s\n", cmd.c_str());
        fflush(pipe);
    }
private:
    FILE* pipe;
};

int main() {
    const std::string db_host   = "tcp://127.0.0.1:3306";
    const std::string db_user   = "car_user";
    const std::string db_pass   = "StrongPassword123!";
    const std::string db_schema = "car_tracking";

    try {
        sql::mysql::MySQL_Driver* driver = sql::mysql::get_mysql_driver_instance();
        std::unique_ptr<sql::Connection> con(driver->connect(db_host, db_user, db_pass));
        con->setSchema(db_schema);
        std::unique_ptr<sql::Statement> stmt(con->createStatement());

        /* 读取轨迹 */
        std::map<int, std::vector<Point>> trajectories;
        std::unique_ptr<sql::ResultSet> res(stmt->executeQuery(
            "SELECT marker_id, world_x, world_y, world_theta, timestamp "
            "FROM car_positions ORDER BY marker_id, timestamp ASC"));

        while (res->next()) {
            int id = res->getInt("marker_id");
            trajectories[id].push_back({
                res->getDouble("world_x"),
                res->getDouble("world_y"),
                res->getDouble("world_theta"),
                res->getDouble("timestamp")
            });
        }

        if (trajectories.empty()) {
            std::cout << "数据库中无轨迹数据。" << std::endl;
            return 0;
        }

        /* 写临时数据文件并绘图 */
        GnuplotPipe gp;
        gp.send("set title '小车轨迹 (world_x vs world_y)'");
        gp.send("set xlabel 'world_x [m]'");
        gp.send("set ylabel 'world_y [m]'");
        gp.send("set grid");
        gp.send("set key outside right");

        std::string plotCmd = "plot ";
        int idx = 0;
        for (const auto& [id, pts] : trajectories) {
            if (pts.empty()) continue;
            std::string fname = "traj_" + std::to_string(id) + ".dat";
            std::ofstream ofs(fname);
            for (const auto& p : pts)
                ofs << std::fixed << std::setprecision(3)
                    << p.x << " " << p.y << "\n";
            ofs.close();

            if (idx++) plotCmd += ", ";
            plotCmd += "'" + fname + "' using 1:2 with lines lw 2 title 'ID " + std::to_string(id) + "'";
        }
        gp.send(plotCmd);

        std::cout << "轨迹已绘制，按 Enter 键退出..." << std::endl;
        std::cin.get();

        /* 清理临时文件 */
        for (const auto& [id, _] : trajectories)
            std::remove(("traj_" + std::to_string(id) + ".dat").c_str());

    } catch (sql::SQLException& e) {
        std::cerr << "SQL Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}