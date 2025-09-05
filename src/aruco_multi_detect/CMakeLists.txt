cmake_minimum_required(VERSION 3.0.2)
project(aruco_multi_detect)
set(CMAKE_CXX_STANDARD 17)
# --- 查找依赖包 ---
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  image_transport
  cv_bridge
  message_generation
  message_runtime
  message_filters
)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc aruco) # 明确指定OpenCV组件，确保aruco模块被找到
find_package(OpenMP) # 查找 OpenMP
find_package(Eigen3 REQUIRED) # 查找 Eigen3
find_package(Boost REQUIRED COMPONENTS system) # 查找 Boost system 组件

# --- Qt5 ---
find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui Concurrent)
# 移除 find_package(QCustomPlot REQUIRED)，因为 apt-get 安装的版本不提供该功能
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# --- MySQL Connector/C++ 配置开始 ---
# 查找 MySQL Connector/C++ 头文件和库
find_path(MYSQLCPPCONN_INCLUDE_DIR mysql_driver.h
  PATHS
  /usr/include
  /usr/include/cppconn
  /usr/local/include/cppconn
  /usr/include/mysql
  /usr/local/include/mysql
  /opt/mysql/mysql-connector-c++/include
  NO_DEFAULT_PATH
)

# 查找 libmysqlcppconn 库
find_library(MYSQLCPPCONN_LIBRARY mysqlcppconn
  PATHS
  /usr/lib/aarch64-linux-gnu
  /usr/lib/x86_64-linux-gnu
  /usr/lib
  /usr/local/lib
  /opt/mysql/mysql-connector-c++/lib
  NO_DEFAULT_PATH
)

# 查找 mysqlclient C API 库 (mysqlcppconn 内部可能依赖它)
find_library(MYSQLCLIENT_LIBRARY mysqlclient
  PATHS
  /usr/lib/aarch64-linux-gnu
  /usr/lib/x86_64-linux-gnu
  /usr/lib
  /usr/local/lib
  NO_DEFAULT_PATH
)

if(MYSQLCPPCONN_INCLUDE_DIR AND MYSQLCPPCONN_LIBRARY)
  message(STATUS "Found MySQL Connector/C++ headers: ${MYSQLCPPCONN_INCLUDE_DIR}")
  message(STATUS "Found MySQL Connector/C++ library: ${MYSQLCPPCONN_LIBRARY}")
  set(MYSQL_LIBRARIES ${MYSQLCPPCONN_LIBRARY}) # 主库
  if(MYSQLCLIENT_LIBRARY)
      list(APPEND MYSQL_LIBRARIES ${MYSQLCLIENT_LIBRARY}) # 如果找到，也链接 C API 库
  endif()
  list(APPEND MYSQL_LIBRARIES pthread) # std::thread 或 mysqlcppconn 可能依赖 pthread
else()
  message(FATAL_ERROR "Could not find MySQL Connector/C++ libraries. Please install it or adjust search PATHS in CMakeLists.txt.")
endif()
# --- MySQL Connector/C++ 配置结束 ---

# OpenMP 编译标志
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# --- 消息定义 ---
add_message_files(
  FILES
  ArUcoMarker.msg
  ArUcoMarkers.msg
  NashEquilibrium.msg
  VirtualLeader.msg
  TwistStamped.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

# --- 包配置 ---
catkin_package(
  CATKIN_DEPENDS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  message_runtime
  geometry_msgs
  message_filters
)

# --- 包含目录 ---
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${MYSQLCPPCONN_INCLUDE_DIR}
  ${Qt5_INCLUDE_DIRS}
  /usr/include/qcustomplot # 新增: QCustomPlot头文件路径
  include
  /opt/ros/noetic/include 
)

# --- 可执行文件 ---
# aruco_multi_detector
add_executable(aruco_multi_detector
  src/aruco_multi_detector.cpp
)
add_dependencies(aruco_multi_detector ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(aruco_multi_detector
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${MYSQL_LIBRARIES}
)
if(OPENMP_FOUND)
    target_link_libraries(aruco_multi_detector OpenMP::OpenMP_CXX)
endif()

# pid_controller
add_executable(pid_controller
  src/pid_controller.cpp
)
target_link_libraries(pid_controller
  ${catkin_LIBRARIES}
)

# robots_control
add_executable(robots_control src/robots_control.cpp)
add_dependencies(robots_control ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(robots_control
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# mysql_write
add_executable(mysql_write src/mysql_write.cpp)
target_link_libraries(mysql_write
  ${catkin_LIBRARIES}
  ${MYSQL_LIBRARIES}
)

# mysql_test
add_executable(mysql_test src/mysql_test.cpp)
target_link_libraries(mysql_test
  ${catkin_LIBRARIES}
)



# mysql_read (Qt Application)
add_executable(mysql_read
  src/mysql_read.cpp
)
target_link_libraries(mysql_read
  ${catkin_LIBRARIES}
  ${MYSQL_LIBRARIES}
  ${Boost_LIBRARIES}
  Qt5::Concurrent
  Qt5::Core
  Qt5::Gui
  Qt5::Widgets
  qcustomplot # 修改: 直接链接库名
)

# # CMake 最低版本要求
# cmake_minimum_required(VERSION 3.0.2)
# project(aruco_multi_detect)
# set(CMAKE_CXX_STANDARD 17)
# # --- 查找依赖包 ---
# find_package(catkin REQUIRED COMPONENTS
#   roscpp
#   std_msgs
#   geometry_msgs
#   sensor_msgs
#   image_transport
#   cv_bridge
#   message_generation
#   message_runtime
#   message_filters
# )

# find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc aruco) # 明确指定OpenCV组件，确保aruco模块被找到
# find_package(OpenMP) # 查找 OpenMP
# find_package(Eigen3 REQUIRED) # 查找 Eigen3
# find_package(Boost REQUIRED COMPONENTS system) # 查找 Boost system 组件

# # --- Qt5 ---
# find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui)
# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTORCC ON)
# set(CMAKE_AUTOUIC ON)

# # --- MySQL Connector/C++ 配置开始 ---
# # 查找 MySQL Connector/C++ 头文件和库
# find_path(MYSQLCPPCONN_INCLUDE_DIR mysql_driver.h
#   PATHS
#   /usr/include
#   /usr/include/cppconn
#   /usr/local/include/cppconn
#   /usr/include/mysql
#   /usr/local/include/mysql
#   /opt/mysql/mysql-connector-c++/include
#   NO_DEFAULT_PATH
# )

# # 查找 libmysqlcppconn 库
# find_library(MYSQLCPPCONN_LIBRARY mysqlcppconn
#   PATHS
#   /usr/lib/aarch64-linux-gnu
#   /usr/lib/x86_64-linux-gnu
#   /usr/lib
#   /usr/local/lib
#   /opt/mysql/mysql-connector-c++/lib
#   NO_DEFAULT_PATH
# )

# # 查找 mysqlclient C API 库 (mysqlcppconn 内部可能依赖它)
# find_library(MYSQLCLIENT_LIBRARY mysqlclient
#   PATHS
#   /usr/lib/aarch64-linux-gnu
#   /usr/lib/x86_64-linux-gnu
#   /usr/lib
#   /usr/local/lib
#   NO_DEFAULT_PATH
# )

# if(MYSQLCPPCONN_INCLUDE_DIR AND MYSQLCPPCONN_LIBRARY)
#   message(STATUS "Found MySQL Connector/C++ headers: ${MYSQLCPPCONN_INCLUDE_DIR}")
#   message(STATUS "Found MySQL Connector/C++ library: ${MYSQLCPPCONN_LIBRARY}")
#   set(MYSQL_LIBRARIES ${MYSQLCPPCONN_LIBRARY}) # 主库
#   if(MYSQLCLIENT_LIBRARY)
#       list(APPEND MYSQL_LIBRARIES ${MYSQLCLIENT_LIBRARY}) # 如果找到，也链接 C API 库
#   endif()
#   list(APPEND MYSQL_LIBRARIES pthread) # std::thread 或 mysqlcppconn 可能依赖 pthread
# else()
#   message(FATAL_ERROR "Could not find MySQL Connector/C++ libraries. Please install it or adjust search PATHS in CMakeLists.txt.")
# endif()
# # --- MySQL Connector/C++ 配置结束 ---

# # OpenMP 编译标志
# if(OPENMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

# # --- 消息定义 ---
# add_message_files(
#   FILES
#   ArUcoMarker.msg
#   ArUcoMarkers.msg
#   NashEquilibrium.msg
#   VirtualLeader.msg
#   TwistStamped.msg # <-- 新增行
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   geometry_msgs # <-- 确保这里有 geometry_msgs，因为 TwistStamped 依赖它
# )

# # --- 包配置 ---
# catkin_package(
#   # INCLUDE_DIRS include # 如果有自定义头文件安装到包的 include 目录，可以取消注释
#   # LIBRARIES aruco_multi_detect # 如果构建了库，可以取消注释
#   CATKIN_DEPENDS
#   roscpp
#   std_msgs
#   sensor_msgs
#   image_transport
#   cv_bridge
#   message_runtime
#   geometry_msgs
#   message_filters
#   # 添加 mysql_test 和 mysql_write 依赖的外部库
#   # 注意：mysqlcppconn不是catkin包，所以不能放在CATKIN_DEPENDS里，而是作为目标链接库
# )

# # --- 包含目录 ---
# include_directories(
#   # order matters for some compilers
#   ${catkin_INCLUDE_DIRS}
#   ${OpenCV_INCLUDE_DIRS}
#   ${EIGEN3_INCLUDE_DIRS}
#   ${MYSQLCPPCONN_INCLUDE_DIR}
#   ${Qt5_INCLUDE_DIRS}
#   include # 你的项目本地头文件目录
#   # 如果 /opt/ros/noetic/include 不在 catkin_INCLUDE_DIRS 里，可以明确添加
#   /opt/ros/noetic/include 
# )

# # --- 可执行文件 ---
# # aruco_multi_detector
# add_executable(aruco_multi_detector
#   src/aruco_multi_detector.cpp
# )
# target_link_libraries(aruco_multi_detector
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${Boost_LIBRARIES}
#   ${MYSQL_LIBRARIES} # 链接MySQL库
# )
# if(OPENMP_FOUND)
#     target_link_libraries(aruco_multi_detector OpenMP::OpenMP_CXX)
# endif()

# # pid_controller
# add_executable(pid_controller
#   src/pid_controller.cpp
# )
# target_link_libraries(pid_controller
#   ${catkin_LIBRARIES}
# )

# # robots_control
# add_executable(robots_control src/robots_control.cpp)
# target_link_libraries(robots_control
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES} # 这里的OpenCV_LIBRARIES可能需要更明确的组件，如 OpenCV::core OpenCV::highgui (如果需要)
# )

# # mysql_write
# add_executable(mysql_write src/mysql_write.cpp)
# target_link_libraries(mysql_write
#   ${catkin_LIBRARIES}
#   ${MYSQL_LIBRARIES}
# )

# # mysql_test
# add_executable(mysql_test src/mysql_test.cpp)
# target_link_libraries(mysql_test
#   ${catkin_LIBRARIES}
# )

# # db_test_and_plot (如果你希望编译这个，它没有在之前的错误中出现)
# add_executable(db_test_and_plot src/db_test_and_plot.cpp)
# target_link_libraries(db_test_and_plot
#   ${catkin_LIBRARIES} # 可能不需要catkin，但为了统一先加上
#   ${MYSQL_LIBRARIES}
# )

# # plot_from_db (如果你希望编译这个)
# add_executable(plot_from_db src/plot_from_db.cpp)
# target_link_libraries(plot_from_db
#   ${catkin_LIBRARIES} # 可能不需要catkin，但为了统一先加上
#   ${MYSQL_LIBRARIES}
# )

# # mysql_read (Qt Application)
# add_executable(mysql_read
#   src/mysql_read.cpp
#   src/qcustomplot.cpp
# )
# target_link_libraries(mysql_read
#   ${catkin_LIBRARIES}
#   #${MYSQL_LIBRARIES}
#   ${QT_LIBRARIES}
#   {Boost_LIBRARIES}
#   Qt5::Core
#   Qt5::Gui
#   Qt5::Widgets
# )



# # --- 安装目标 ---
# install(TARGETS
#   aruco_multi_detector
#   pid_controller
#   robots_control
#   mysql_write
#   mysql_test
#   db_test_and_plot # 添加安装目标
#   plot_from_db     # 添加安装目标
#   mysql_read
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(DIRECTORY include/${PROJECT_NAME}/ # 假设你有头文件在此目录
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# )
