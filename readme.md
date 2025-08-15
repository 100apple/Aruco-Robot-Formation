当前版本0.1.3
优化mysql_test数据，优化mysql_read绘图，增加python可视化工具

之前更新：
0.1.2
增加mysql_read可视化界面功能
0.1.0：
功能解耦，新增qt绘图

绘图测试使用方法：
(确保启动数据库服务)
1.roscore
2.运行mysql_test.cpp
3.运行mysql_write.cpp
4.运行mysql_read.cpp

使用说明：
aruco_multi_detect.cpp 识别小车位姿并写入数据库
db_test_and_plot.cpp 测试数据库绘图（暂不用）
plot_test.cpp 数据库画图
robot_control.cpp   多机器人控制
pid_controller 控制测试
mysql_test.cpp 生成虚拟数据发布到话题
mysql_write.cpp 接受aruco_multi_detect.cpp与robot_control.cpp话题并写入数据库
mysql_read.cpp 读取数据库并绘图


需要安装：
sudo apt-get install libqcustomplot-dev
sudo apt-get install qtbase5-dev
sudo apt-get install libqt5svg5-dev
sudo apt install gnuplot
sudo apt-get install libmysqlcppconn-dev
