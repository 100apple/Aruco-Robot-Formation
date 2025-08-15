#include <ros/ros.h>

#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QComboBox>
#include <QMessageBox>
#include <QTabWidget>
#include <QSpinBox>
#include <QDateTimeEdit>
#include <QColor>
#include <QSlider>
#include <QCheckBox>
#include <QGroupBox>
#include <QTimer>
#include <QDebug>
#include <QThread>
#include <QProgressBar>
#include <QStatusBar>
#include <QMap>

// MUST include QMetaType for qRegisterMetaType
#include <QMetaType>

#include <thread>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <limits> // For std::numeric_limits
#include <memory>

// MySQL Connector/C++
// Ensure these paths are correct for your system
#include <mysql_driver.h>
#include <mysql_connection.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

// Include QCustomPlot source directly if not using a precompiled library
// This needs to be available in the same directory or adjust include path in CMakeLists.txt.
#include "qcustomplot.h"

// =========================================================================
// LTTB (Largest-Triangle-Three-Buckets) Downsampling Algorithm
// =========================================================================

// A generic point structure for LTTB, decoupled from PlotDataPoint for reusability.
struct LTTBPoint {
    double x, y;
};

// LTTB (Largest-Triangle-Three-Buckets) Downsampling Algorithm
// @param data: The original data points.
// @param threshold: The number of data points to return.
// @param getX: A lambda/function pointer to extract the X-coordinate from a data point.
// @param getY: A lambda/function pointer to extract the Y-coordinate from a data point.
// @return: A downsampled version of the data.
template<typename T>
QVector<T> downsampleLTTB(const QVector<T>& data, int threshold,
                         double (*getX)(const T&), double (*getY)(const T&)) {
    if (threshold >= data.size() || threshold <= 0) {
        return data; // Return original data if threshold is invalid or not needed
    }

    QVector<T> sampled;
    sampled.reserve(threshold);

    if (data.size() < 3) { // Need at least first, last, and one intermediate point.
        return data; // Not enough points to properly downsample with LTTB.
    }

    // Always add the first point
    sampled.append(data.first());

    // Calculate effective bucket size. We need threshold - 2 points from the middle.
    double every = static_cast<double>(data.size() - 2) / (threshold - 2);

    int a = 0; // Index of the anchor point (previous point that was added to sampled)

    for (int i = 0; i < threshold - 2; ++i) {
        // Calculate the range for the current bucket
        int bucket_start = static_cast<int>(std::floor((i + 0) * every)) + 1;
        int bucket_end = static_cast<int>(std::floor((i + 1) * every)) + 1;
        // Ensure bucket_end does not exceed data bounds (can happen for last bucket)
        // Upper bound for bucket_end should be `data.size() - 1` (last point can't be in a bucket for calculation)
        // or just `data.size()` as loop `j < bucket_end`
        if (bucket_end >= data.size()) { // Cap at data.size() - 1 for proper indexing
            bucket_end = data.size() - 1;
        }
        if (bucket_start >= bucket_end) { // Avoid empty buckets or invalid ranges
            // This can happen if 'every' is too small, or threshold is close to data.size()
            // In this case, just pick the nearest point logically, or skip.
            continue; // Skip this bucket if it's invalid
        }


        // The "next" point for triangle calculation (point C in ABC triangle).
        // It's conceptually the point that belongs to the *next* bucket.
        int point_c_candidate_index = static_cast<int>(std::floor((i + 2) * every)) + 1;
        if (point_c_candidate_index >= data.size()) {
            point_c_candidate_index = data.size() - 1; // Ensure it refers to the last data point if out of bounds
        }
        
        // Define points A and C for triangle area calculation
        double point_a_x = getX(data[a]);
        double point_a_y = getY(data[a]);
        double point_c_x = getX(data[point_c_candidate_index]);
        double point_c_y = getY(data[point_c_candidate_index]);

        double max_area = -1.0;
        int best_point_index = -1;

        // Iterate through the current bucket to find the point that forms the largest triangle
        // with the anchor point (A) and the candidate for the next point (C).
        for (int j = bucket_start; j < bucket_end; ++j) {
            double point_b_x = getX(data[j]);
            double point_b_y = getY(data[j]);
             
            // Area = 0.5 * |Ax(By - Cy) + Bx(Cy - Ay) + Cx(Ay - By)|
            // We can omit the 0.5 as we only care about relative area (which one is largest)
            double area = std::abs(point_a_x * (point_b_y - point_c_y) +
                                   point_b_x * (point_c_y - point_a_y) +
                                   point_c_x * (point_a_y - point_b_y));

            if (area > max_area) {
                max_area = area;
                best_point_index = j;
            }
        }
        
        if (best_point_index != -1) {
            sampled.append(data[best_point_index]);
            a = best_point_index; // Update the anchor point for the next bucket
        } else {
            // This case might happen if a bucket results in bucket_start >= bucket_end,
            // or if all points in a bucket produced 0 area.
            // To ensure progress, it's safer to just pick the last point of the previous bucket
            // or the first point of the current bucket if best_point_index remains -1.
            // For robust LTTB, best_point_index should almost always be found if bucket is valid.
            // If it still happens, it indicates an edge case with very few points or specific data patterns.
            qWarning() << "LTTB: No best point found in bucket. Bucket [" << bucket_start << "," << bucket_end << "). Anchor:" << a << "C-candidate:" << point_c_candidate_index;
            // Fallback: If no best point found, take the first point of the bucket to ensure progression.
            if (bucket_start < data.size() && bucket_start < point_c_candidate_index) {
                sampled.append(data[bucket_start]);
                a = bucket_start;
            }
        }
    }

    // Always add the last point
    sampled.append(data.last());
    return sampled;
}

// =========================================================================
// Data Structures
// =========================================================================

// 用于存储从数据库单行读取的完整数据结构
struct PlotDataPoint {
    double timestamp;
    double actual_x, actual_y;
    double nash_x, nash_y;
    double linear_vel, angular_vel;
};

// Define a type alias for the complex map for Q_DECLARE_METATYPE
using RobotDataMap = std::map<int, QVector<PlotDataPoint>>;

// Q_DECLARE_METATYPE declarations must be at global scope and before QApplication is created
// if used in signals/slots across threads. This location is appropriate.
Q_DECLARE_METATYPE(PlotDataPoint)
Q_DECLARE_METATYPE(QVector<PlotDataPoint>)
Q_DECLARE_METATYPE(RobotDataMap)

// =========================================================================
// Database Worker Class (runs in a separate thread)
// =========================================================================

class DatabaseWorker : public QObject {
    Q_OBJECT

public:
    DatabaseWorker() : m_driver(nullptr) {
        qDebug() << "DatabaseWorker created";
    }

    ~DatabaseWorker() {
        qDebug() << "DatabaseWorker destroyed";
        if (m_con) {
            try {
                m_con->close();
            } catch (const sql::SQLException& e) {
                qDebug() << "Error closing connection: " << e.what();
            }
        }
    }

public slots:
    void connectToDatabase(const QString& host, const QString& user,
                          const QString& password, const QString& db) {
        qDebug() << "Connecting to database in worker thread";
        try {
            m_driver = sql::mysql::get_mysql_driver_instance();
            if (!m_driver) {
                emit connectionResult(false, "无法获取MySQL驱动实例");
                return;
            }

            m_con.reset();
            m_con.reset(m_driver->connect(host.toStdString(),
                                         user.toStdString(),
                                         password.toStdString()));

            if (!m_con) {
                emit connectionResult(false, "数据库连接失败");
                return;
            }

            m_con->setSchema(db.toStdString());
            emit connectionResult(true, "数据库连接成功！");
            qDebug() << "Database connection successful in worker thread";

        } catch (sql::SQLException &e) {
            QString errorMsg = QString("连接失败: %1\n(错误码: %2, SQLState: %3)")
                .arg(e.what()).arg(e.getErrorCode()).arg(e.getSQLState().c_str());
            emit connectionResult(false, errorMsg);
            m_con.reset();
            qDebug() << "Database connection failed:" << e.what();
        }
    }

    void loadTableNames() {
        qDebug() << "Loading table names in worker thread";
        if (!m_con || m_con->isClosed()) {
            emit tableNamesLoaded(QStringList());
            return;
        }

        try {
            std::unique_ptr<sql::Statement> stmt(m_con->createStatement());
            std::unique_ptr<sql::ResultSet> res(stmt->executeQuery("SHOW TABLES"));

            QStringList tables;
            tables.append("");
            while (res->next()) {
                tables.append(QString::fromStdString(res->getString(1)));
            }

            emit tableNamesLoaded(tables);
            qDebug() << "Table names loaded in worker thread";

        } catch (sql::SQLException &e) {
            qDebug() << "Failed to load table names:" << e.what();
            emit tableNamesLoaded(QStringList());
        }
    }

    void getTimeRange(const QString& tableName) {
        qDebug() << "Getting time range in worker thread for table:" << tableName;
        if (!m_con || m_con->isClosed() || tableName.isEmpty()) {
            emit timeRangeResult(0, 0, false);
            return;
        }

        try {
            std::unique_ptr<sql::Statement> stmt(m_con->createStatement());
            std::unique_ptr<sql::ResultSet> res(stmt->executeQuery(
                "SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM `" +
                tableName.toStdString() + "` WHERE timestamp IS NOT NULL"));

            if (res->next()) {
                double min_ts = res->getDouble("min_ts");
                double max_ts = res->getDouble("max_ts");

                if (res->isNull("min_ts") || res->isNull("max_ts") || (min_ts == 0.0 && max_ts == 0.0 && res->rowsCount() == 0)) {
                    emit timeRangeResult(0, 0, false);
                    qDebug() << "Table is empty or timestamp column has no data for min/max.";
                    return;
                }

                emit timeRangeResult(min_ts, max_ts, true);
                qDebug() << QString("Time range loaded in worker thread: [%1, %2]").arg(min_ts).arg(max_ts);
            } else {
                emit timeRangeResult(0, 0, false);
                qDebug() << "No time range data found (empty result set for min/max query).";
            }

        } catch (sql::SQLException &e) {
            qDebug() << "Failed to get time range (SQL Error):" << e.what();
            emit timeRangeResult(0, 0, false);
        }
    }

    void queryData(const QString& tableName, qint64 start_ts, qint64 end_ts) {
        qDebug() << QString("Querying data in worker thread for table '%1' from %2 to %3")
                    .arg(tableName).arg(start_ts).arg(end_ts);
        RobotDataMap robotDataCache;

        if (tableName.isEmpty() || !m_con || m_con->isClosed()) {
            emit dataQueryResult(robotDataCache, false);
            return;
        }

        try {
            std::unique_ptr<sql::PreparedStatement> pstmt(m_con->prepareStatement(
                "SELECT timestamp, robot_id, actual_x, actual_y, nash_x, nash_y, linear_vel, angular_vel "
                "FROM `" + tableName.toStdString() + "`"
                " WHERE timestamp BETWEEN ? AND ? ORDER BY robot_id, timestamp ASC"
            ));

            pstmt->setDouble(1, static_cast<double>(start_ts));
            pstmt->setDouble(2, static_cast<double>(end_ts));

            std::unique_ptr<sql::ResultSet> res(pstmt->executeQuery());

            int row_count = 0;
            // No longer checking res->supports->rowsCount().
            // Call rowsCount() directly. If it's a streaming result, it may be 0 or -1.
            size_t estimated_total_rows = res->rowsCount();
            qDebug() << "Estimated rows to fetch:" << estimated_total_rows;


            while(res->next()){
                row_count++;
                int id = res->getInt("robot_id");

                if (estimated_total_rows > 0 && robotDataCache.find(id) == robotDataCache.end()) {
                    robotDataCache[id].reserve(estimated_total_rows / std::max(1, (int)robotDataCache.size()));
                }

                PlotDataPoint point = {
                    res->getDouble("timestamp"),
                    res->getDouble("actual_x"),
                    res->getDouble("actual_y"),
                    res->getDouble("nash_x"),
                    res->getDouble("nash_y"),
                    res->getDouble("linear_vel"),
                    res->getDouble("angular_vel")
                };
                robotDataCache[id].append(point);
            }

            qDebug() << "Query returned " << row_count << " rows.";
            emit dataQueryResult(robotDataCache, true);

        } catch (sql::SQLException &e) {
            QString errorMsg = QString("数据库查询失败: %1\n(错误码: %2, SQLState: %3)")
                .arg(e.what()).arg(e.getErrorCode()).arg(e.getSQLState().c_str());
            qDebug() << errorMsg;
            emit dataQueryResult(robotDataCache, false);
        }
    }

signals:
    void connectionResult(bool success, const QString& message);
    void tableNamesLoaded(const QStringList& tables);
    void timeRangeResult(double min_ts, double max_ts, bool success);
    void dataQueryResult(const RobotDataMap& data, bool success);

private:
    sql::mysql::MySQL_Driver* m_driver;
    std::unique_ptr<sql::Connection> m_con;
};

// =========================================================================
// Main Window Class
// =========================================================================

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

signals:
    void requestConnectToDatabase(const QString& host, const QString& user, const QString& password, const QString& db);
    void requestLoadTableNames();
    void requestGetTimeRange(const QString& tableName);
    void requestQueryData(const QString& tableName, qint64 start_ts, qint64 end_ts);

private slots:
    void connectToDatabase();
    void handleConnectionResult(bool success, const QString& message);
    void handleTableNamesLoaded(const QStringList& tables);
    void handleTimeRangeResult(double min_ts, double max_ts, bool success);
    void handleDataQueryResult(const RobotDataMap& data, bool success);
    void onPlotStylesChanged();
    void fetchAndPlotData();
    void onStartRangeSliderValueChanged(int value);
    void onEndRangeSliderValueChanged(int value);
    void onLiveUpdateToggled(int state);
    void onTableNameChanged(const QString& tableName);

private:
    void setupUi();
    void setupConnectionWidgets(QLayout *layout);
    void setupControls(QLayout* layout);
    QCustomPlot* createPlotTab(const QString& title, QCPTextElement*& titleElementRef,
                              QLineEdit* titleEditPtr, QLineEdit* xLabelEditPtr, QLineEdit* yLabelEditPtr);
    void plotAll();
    void plotTrajectories();
    void plotVelocities();
    void plotErrors();
    void updatePlotTextAndStyles(QCustomPlot* plot, QCPTextElement* titleElement,
                            QLineEdit* titleEdit, QLineEdit* xLabelEdit, QLineEdit* yLabelEdit);

    void clearAllPlots();
    void setTimeControlsEnabled(bool enabled);
    void showStatusMessage(const QString& message, int timeout = 0);
    void setLoadingState(bool loading);

    QLineEdit *m_hostLineEdit, *m_userLineEdit, *m_passLineEdit, *m_dbLineEdit;
    QPushButton *m_connectButton, *m_nowButton;
    QComboBox *m_tableComboBox;
    QDateTimeEdit *m_startTimeEdit, *m_endTimeEdit;
    QPushButton *m_plotButton;
    QSpinBox *m_lineWidthSpinBox;
    QTabWidget *m_tabWidget;
    QProgressBar *m_progressBar;
    QStatusBar *m_statusBar;

    QSlider *m_startRangeSlider;
    QSlider *m_endRangeSlider;
    QCheckBox *m_liveUpdateCheckBox;
    QTimer *m_updateTimer;
    QSpinBox *m_titleSizeSpinBox;
    QCheckBox *m_titleBoldCheckBox;
    QSpinBox *m_labelSizeSpinBox;
    QCheckBox *m_labelBoldCheckBox;
    QLineEdit *m_trajectoryTitleEdit, *m_velocityTitleEdit, *m_errorTitleEdit;
    QLineEdit *m_trajectoryXLabelEdit, *m_trajectoryYLabelEdit;
    QLineEdit *m_velocityXLabelEdit, *m_velocityYLabelEdit;
    QLineEdit *m_errorXLabelEdit, *m_errorYLabelEdit;

    QCustomPlot *m_trajectoryPlot;
    QCustomPlot *m_velocityPlot;
    QCustomPlot *m_errorPlot;
    QCPTextElement *m_trajectoryTitleElement = nullptr;
    QCPTextElement *m_velocityTitleElement = nullptr;
    QCPTextElement *m_errorTitleElement = nullptr;

    QThread m_dbThread;
    DatabaseWorker m_dbWorker;

    RobotDataMap m_robotDataCache;

    double m_minTableTimestamp = 0;
    double m_maxTableTimestamp = 0;

    const int SLIDER_RESOLUTION = 10000;
    const int PLOT_DOWNSAMPLE_THRESHOLD = 4000;
};

// =========================================================================
// Main Window Implementations
// =========================================================================

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    m_dbWorker.moveToThread(&m_dbThread);
    connect(&m_dbThread, &QThread::finished, &m_dbWorker, &QObject::deleteLater);
    m_dbThread.start();

    connect(this, &MainWindow::requestConnectToDatabase, &m_dbWorker, &DatabaseWorker::connectToDatabase);
    connect(&m_dbWorker, &DatabaseWorker::connectionResult, this, &MainWindow::handleConnectionResult);
    connect(this, &MainWindow::requestLoadTableNames, &m_dbWorker, &DatabaseWorker::loadTableNames);
    connect(&m_dbWorker, &DatabaseWorker::tableNamesLoaded, this, &MainWindow::handleTableNamesLoaded);
    connect(this, &MainWindow::requestGetTimeRange, &m_dbWorker, &DatabaseWorker::getTimeRange);
    connect(&m_dbWorker, &DatabaseWorker::timeRangeResult, this, &MainWindow::handleTimeRangeResult);
    connect(this, &MainWindow::requestQueryData, &m_dbWorker, &DatabaseWorker::queryData);
    connect(&m_dbWorker, &DatabaseWorker::dataQueryResult, this, &MainWindow::handleDataQueryResult);

    setupUi();

    connect(m_connectButton, &QPushButton::clicked, this, &MainWindow::connectToDatabase);
    connect(m_plotButton, &QPushButton::clicked, this, &MainWindow::fetchAndPlotData);
    connect(m_lineWidthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onPlotStylesChanged);
    connect(m_titleSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onPlotStylesChanged);
    connect(m_titleBoldCheckBox, &QCheckBox::stateChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_labelSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onPlotStylesChanged);
    connect(m_labelBoldCheckBox, &QCheckBox::stateChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_trajectoryTitleEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_trajectoryXLabelEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_trajectoryYLabelEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_velocityTitleEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_velocityXLabelEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_velocityYLabelEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_errorTitleEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_errorXLabelEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_errorYLabelEdit, &QLineEdit::textChanged, this, &MainWindow::onPlotStylesChanged);
    connect(m_startRangeSlider, &QSlider::valueChanged, this, &MainWindow::onStartRangeSliderValueChanged);
    connect(m_endRangeSlider, &QSlider::valueChanged, this, &MainWindow::onEndRangeSliderValueChanged);
    connect(m_tableComboBox, QOverload<const QString&>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onTableNameChanged);
    connect(m_nowButton, &QPushButton::clicked, [this]() {
        m_endTimeEdit->setDateTime(QDateTime::currentDateTime());
        if (!m_liveUpdateCheckBox->isChecked()) {
             m_endRangeSlider->setValue(SLIDER_RESOLUTION);
        }
    });

    connect(m_liveUpdateCheckBox, &QCheckBox::stateChanged, this, &MainWindow::onLiveUpdateToggled);
    m_updateTimer = new QTimer(this);
    m_updateTimer->setInterval(1000);
    connect(m_updateTimer, &QTimer::timeout, this, &MainWindow::fetchAndPlotData);

    setTimeControlsEnabled(false);
    onPlotStylesChanged();
}

MainWindow::~MainWindow() {
    if (m_updateTimer && m_updateTimer->isActive()) {
        m_updateTimer->stop();
    }
    m_dbThread.quit();
    m_dbThread.wait(5000);

    clearAllPlots();
}

void MainWindow::setupUi() {
    setWindowTitle("MySQL 数据可视化分析工具");
    resize(1600, 900);

    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    m_statusBar = new QStatusBar(this);
    setStatusBar(m_statusBar);
    m_progressBar = new QProgressBar(m_statusBar);
    m_progressBar->setMaximumWidth(200);
    m_progressBar->setTextVisible(false);
    m_progressBar->setVisible(false);
    m_statusBar->addPermanentWidget(m_progressBar);

    QWidget *topPanel = new QWidget();
    topPanel->setMaximumHeight(200);
    QGridLayout *topLayout = new QGridLayout(topPanel);
    setupConnectionWidgets(topLayout);
    setupControls(topLayout);
    mainLayout->addWidget(topPanel);

    m_tabWidget = new QTabWidget();
    mainLayout->addWidget(m_tabWidget);

    m_trajectoryPlot = createPlotTab("轨迹图 (Trajectory)", m_trajectoryTitleElement,
                                    m_trajectoryTitleEdit, m_trajectoryXLabelEdit, m_trajectoryYLabelEdit);
    m_velocityPlot = createPlotTab("速度图 (Velocity)", m_velocityTitleElement,
                                  m_velocityTitleEdit, m_velocityXLabelEdit, m_velocityYLabelEdit);
    m_errorPlot = createPlotTab("误差图 (Error)", m_errorTitleElement,
                               m_errorTitleEdit, m_errorXLabelEdit, m_errorYLabelEdit);
}

void MainWindow::setupConnectionWidgets(QLayout *layout) {
    auto gridLayout = qobject_cast<QGridLayout*>(layout);
    if (!gridLayout) return;

    m_hostLineEdit = new QLineEdit("tcp://127.0.0.1:3306");
    m_userLineEdit = new QLineEdit("car_user"); // <-- CHANGE THIS
    m_passLineEdit = new QLineEdit("StrongPassword123!"); // <-- CHANGE THIS
    m_passLineEdit->setEchoMode(QLineEdit::Password);
    m_dbLineEdit = new QLineEdit("car_tracking");
    m_connectButton = new QPushButton("连接数据库");

    QFont labelFont;
    labelFont.setBold(true);

    QLabel* hostLabel = new QLabel("主机:");
    hostLabel->setFont(labelFont);
    gridLayout->addWidget(hostLabel, 0, 0);
    gridLayout->addWidget(m_hostLineEdit, 0, 1);

    QLabel* userLabel = new QLabel("用户:");
    userLabel->setFont(labelFont);
    gridLayout->addWidget(userLabel, 0, 2);
    gridLayout->addWidget(m_userLineEdit, 0, 3);

    QLabel* passLabel = new QLabel("密码:");
    passLabel->setFont(labelFont);
    gridLayout->addWidget(passLabel, 0, 4);
    gridLayout->addWidget(m_passLineEdit, 0, 5);

    QLabel* dbLabel = new QLabel("数据库:");
    dbLabel->setFont(labelFont);
    gridLayout->addWidget(dbLabel, 0, 6);
    gridLayout->addWidget(m_dbLineEdit, 0, 7);

    gridLayout->addWidget(m_connectButton, 0, 8, 1, 2);
}

void MainWindow::setupControls(QLayout* layout) {
    auto gridLayout = qobject_cast<QGridLayout*>(layout);
    if (!gridLayout) return;

    int currentRow = 1;

    gridLayout->addWidget(new QLabel("选择实验数据表:"), currentRow, 0);
    m_tableComboBox = new QComboBox();
    m_tableComboBox->setDisabled(true);
    gridLayout->addWidget(m_tableComboBox, currentRow, 1, 1, 2);

    gridLayout->addWidget(new QLabel("开始时间:"), currentRow, 3);
    m_startTimeEdit = new QDateTimeEdit(QDateTime::fromSecsSinceEpoch(0));
    m_startTimeEdit->setDisplayFormat("yyyy-MM-dd HH:mm:ss");
    m_startTimeEdit->setCalendarPopup(true);
    m_startTimeEdit->setDisabled(true);
    gridLayout->addWidget(m_startTimeEdit, currentRow, 4);

    gridLayout->addWidget(new QLabel("结束时间:"), currentRow, 5);
    m_endTimeEdit = new QDateTimeEdit(QDateTime::currentDateTime());
    m_endTimeEdit->setDisplayFormat("yyyy-MM-dd HH:mm:ss");
    m_endTimeEdit->setCalendarPopup(true);
    m_endTimeEdit->setDisabled(true);
    gridLayout->addWidget(m_endTimeEdit, currentRow, 6);

    m_nowButton = new QPushButton("Now");
    m_nowButton->setDisabled(true);
    gridLayout->addWidget(m_nowButton, currentRow, 7);

    m_liveUpdateCheckBox = new QCheckBox("实时更新");
    m_liveUpdateCheckBox->setDisabled(true);
    gridLayout->addWidget(m_liveUpdateCheckBox, currentRow, 8);
    currentRow++;

    gridLayout->addWidget(new QLabel("开始时间滑块:"), currentRow, 0);
    m_startRangeSlider = new QSlider(Qt::Horizontal);
    m_startRangeSlider->setRange(0, SLIDER_RESOLUTION);
    m_startRangeSlider->setValue(0);
    m_startRangeSlider->setDisabled(true);
    gridLayout->addWidget(m_startRangeSlider, currentRow, 1, 1, 8);
    currentRow++;

    gridLayout->addWidget(new QLabel("结束时间滑块:"), currentRow, 0);
    m_endRangeSlider = new QSlider(Qt::Horizontal);
    m_endRangeSlider->setRange(0, SLIDER_RESOLUTION);
    m_endRangeSlider->setValue(SLIDER_RESOLUTION);
    m_endRangeSlider->setDisabled(true);
    gridLayout->addWidget(m_endRangeSlider, currentRow, 1, 1, 8);
    currentRow++;

    m_plotButton = new QPushButton("开始绘图");
    m_plotButton->setDisabled(true);
    m_plotButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 6px; } QPushButton:hover { background-color: #45a049; }");
    gridLayout->addWidget(m_plotButton, currentRow, 8, 1, 2);
    currentRow++;

    QGroupBox *plotSettingsGroup = new QGroupBox("绘图设置");
    QGridLayout *plotSettingsLayout = new QGridLayout(plotSettingsGroup);

    plotSettingsLayout->addWidget(new QLabel("标题大小:"), 0, 0);
    m_titleSizeSpinBox = new QSpinBox();
    m_titleSizeSpinBox->setRange(8, 24);
    m_titleSizeSpinBox->setValue(12);
    m_titleSizeSpinBox->setDisabled(true);
    plotSettingsLayout->addWidget(m_titleSizeSpinBox, 0, 1);

    m_titleBoldCheckBox = new QCheckBox("粗体");
    m_titleBoldCheckBox->setChecked(true);
    m_titleBoldCheckBox->setDisabled(true);
    plotSettingsLayout->addWidget(m_titleBoldCheckBox, 0, 2);

    plotSettingsLayout->addWidget(new QLabel("标签大小:"), 0, 3);
    m_labelSizeSpinBox = new QSpinBox();
    m_labelSizeSpinBox->setRange(8, 20);
    m_labelSizeSpinBox->setValue(10);
    m_labelSizeSpinBox->setDisabled(true);
    plotSettingsLayout->addWidget(m_labelSizeSpinBox, 0, 4);

    m_labelBoldCheckBox = new QCheckBox("粗体");
    m_labelBoldCheckBox->setChecked(true);
    m_labelBoldCheckBox->setDisabled(true);
    plotSettingsLayout->addWidget(m_labelBoldCheckBox, 0, 5);

    plotSettingsLayout->addWidget(new QLabel("线条粗细:"), 0, 6);
    m_lineWidthSpinBox = new QSpinBox();
    m_lineWidthSpinBox->setRange(1, 10);
    m_lineWidthSpinBox->setValue(2);
    m_lineWidthSpinBox->setDisabled(true);
    plotSettingsLayout->addWidget(m_lineWidthSpinBox, 0, 7);

    int textEditRow = 1;

    plotSettingsLayout->addWidget(new QLabel("轨迹图标题:"), textEditRow, 0);
    m_trajectoryTitleEdit = new QLineEdit("轨迹图 (Trajectory)");
    m_trajectoryTitleEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_trajectoryTitleEdit, textEditRow, 1, 1, 3);
    plotSettingsLayout->addWidget(new QLabel("X轴标签:"), textEditRow, 4);
    m_trajectoryXLabelEdit = new QLineEdit("World X (m)");
    m_trajectoryXLabelEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_trajectoryXLabelEdit, textEditRow, 5);
    plotSettingsLayout->addWidget(new QLabel("Y轴标签:"), textEditRow, 6);
    m_trajectoryYLabelEdit = new QLineEdit("World Y (m)");
    m_trajectoryYLabelEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_trajectoryYLabelEdit, textEditRow, 7);
    textEditRow++;

    plotSettingsLayout->addWidget(new QLabel("速度图标题:"), textEditRow, 0);
    m_velocityTitleEdit = new QLineEdit("速度图 (Velocity)");
    m_velocityTitleEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_velocityTitleEdit, textEditRow, 1, 1, 3);
    plotSettingsLayout->addWidget(new QLabel("X轴标签:"), textEditRow, 4);
    m_velocityXLabelEdit = new QLineEdit("时间 (s)");
    m_velocityXLabelEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_velocityXLabelEdit, textEditRow, 5);
    plotSettingsLayout->addWidget(new QLabel("Y轴标签:"), textEditRow, 6);
    m_velocityYLabelEdit = new QLineEdit("速度 (m/s or rad/s)");
    m_velocityYLabelEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_velocityYLabelEdit, textEditRow, 7);
    textEditRow++;

    plotSettingsLayout->addWidget(new QLabel("误差图标题:"), textEditRow, 0);
    m_errorTitleEdit = new QLineEdit("误差图 (Error)");
    m_errorTitleEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_errorTitleEdit, textEditRow, 1, 1, 3);
    plotSettingsLayout->addWidget(new QLabel("X轴标签:"), textEditRow, 4);
    m_errorXLabelEdit = new QLineEdit("时间 (s)");
    m_errorXLabelEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_errorXLabelEdit, textEditRow, 5);
    plotSettingsLayout->addWidget(new QLabel("Y轴标签:"), textEditRow, 6);
    m_errorYLabelEdit = new QLineEdit("位置误差 (m)");
    m_errorYLabelEdit->setDisabled(true);
    plotSettingsLayout->addWidget(m_errorYLabelEdit, textEditRow, 7);

    gridLayout->addWidget(plotSettingsGroup, currentRow, 0, 1, 10);
}

void MainWindow::connectToDatabase() {
    showStatusMessage("正在连接数据库...");
    setLoadingState(true);

    QString host = m_hostLineEdit->text();
    QString user = m_userLineEdit->text();
    QString pass = m_passLineEdit->text();
    QString db = m_dbLineEdit->text();

    emit requestConnectToDatabase(host, user, pass, db);
}

void MainWindow::handleConnectionResult(bool success, const QString& message) {
    setLoadingState(false);

    if (success) {
        QMessageBox::information(this, "成功", message);

        m_connectButton->setDisabled(true);
        m_hostLineEdit->setDisabled(true);
        m_userLineEdit->setDisabled(true);
        m_passLineEdit->setDisabled(true);
        m_dbLineEdit->setDisabled(true);

        m_tableComboBox->setDisabled(false);
        m_lineWidthSpinBox->setDisabled(false);
        m_titleSizeSpinBox->setDisabled(false);
        m_titleBoldCheckBox->setDisabled(false);
        m_labelSizeSpinBox->setDisabled(false);
        m_labelBoldCheckBox->setDisabled(false);

        m_trajectoryTitleEdit->setDisabled(false);
        m_trajectoryXLabelEdit->setDisabled(false);
        m_trajectoryYLabelEdit->setDisabled(false);
        m_velocityTitleEdit->setDisabled(false);
        m_velocityXLabelEdit->setDisabled(false);
        m_velocityYLabelEdit->setDisabled(false);
        m_errorTitleEdit->setDisabled(false);
        m_errorXLabelEdit->setDisabled(false);
        m_errorYLabelEdit->setDisabled(false);

        showStatusMessage("正在加载数据表列表...");
        setLoadingState(true);
        emit requestLoadTableNames();
    } else {
        QMessageBox::critical(this, "数据库错误", message);
        showStatusMessage("数据库连接失败", 5000);
    }
}

void MainWindow::handleTableNamesLoaded(const QStringList& tables) {
    setLoadingState(false);

    m_tableComboBox->clear();
    m_tableComboBox->addItems(tables);

    if (tables.size() > 1) {
        m_tableComboBox->setCurrentIndex(1);
        showStatusMessage("数据表列表加载完成");
        emit requestGetTimeRange(m_tableComboBox->currentText());
        setLoadingState(true);
    } else {
        showStatusMessage("未找到数据表", 3000);
        setTimeControlsEnabled(false);
    }
}

void MainWindow::handleTimeRangeResult(double min_ts, double max_ts, bool success) {
    setLoadingState(false);

    if (success) {
        m_minTableTimestamp = min_ts;
        m_maxTableTimestamp = max_ts;

        m_startTimeEdit->setDateTime(QDateTime::fromSecsSinceEpoch(static_cast<qint64>(min_ts)));
        m_endTimeEdit->setDateTime(QDateTime::fromSecsSinceEpoch(static_cast<qint64>(max_ts)));

        m_startRangeSlider->setValue(0);
        m_endRangeSlider->setValue(SLIDER_RESOLUTION);

        setTimeControlsEnabled(true);
        showStatusMessage("时间范围加载完成");
    } else {
        setTimeControlsEnabled(false);
        m_minTableTimestamp = 0;
        m_maxTableTimestamp = 0;
        showStatusMessage("无法获取时间范围", 3000);
    }
}

void MainWindow::handleDataQueryResult(const RobotDataMap& data, bool success) {
    setLoadingState(false);

    if (success) {
        m_robotDataCache = data;
        plotAll();
        showStatusMessage("数据加载完成");
    } else {
        m_robotDataCache.clear();
        clearAllPlots();
        showStatusMessage("数据查询失败", 3000);
    }
}

void MainWindow::onTableNameChanged(const QString& tableName) {
    m_minTableTimestamp = 0;
    m_maxTableTimestamp = 0;
    clearAllPlots();
    m_robotDataCache.clear();

    if (tableName.isEmpty()) {
        setTimeControlsEnabled(false);
        return;
    }

    showStatusMessage("正在获取时间范围...");
    setLoadingState(true);
    emit requestGetTimeRange(tableName);
}

void MainWindow::fetchAndPlotData() {
    if (m_liveUpdateCheckBox->isChecked()) {
        m_endTimeEdit->setDateTime(QDateTime::currentDateTime());
    }

    QString tableName = m_tableComboBox->currentText();
    if (tableName.isEmpty() || m_minTableTimestamp == 0 || m_maxTableTimestamp == 0) {
        QMessageBox::warning(this, "提示", "请先选择一个数据表并确保其有有效时间范围。");
        return;
    }

    qint64 start_ts = m_startTimeEdit->dateTime().toSecsSinceEpoch();
    qint64 end_ts = m_endTimeEdit->dateTime().toSecsSinceEpoch();

    if (start_ts > end_ts) {
        QMessageBox::warning(this, "时间范围错误", "开始时间不能晚于结束时间。");
        return;
    }

    showStatusMessage("正在查询数据...");
    setLoadingState(true);
    emit requestQueryData(tableName, start_ts, end_ts);
}

void MainWindow::setLoadingState(bool loading) {
    bool connected = !m_connectButton->isEnabled();

    m_progressBar->setVisible(loading);
    if (loading) {
        m_progressBar->setRange(0, 0);
    } else {
        m_progressBar->setRange(0, 1);
        m_progressBar->setValue(1);
    }

    if (!connected) {
        m_connectButton->setEnabled(!loading);
        m_hostLineEdit->setEnabled(!loading);
        m_userLineEdit->setEnabled(!loading);
        m_passLineEdit->setEnabled(!loading);
        m_dbLineEdit->setEnabled(!loading);
    }

    bool enableDataControls = connected && !loading && (m_maxTableTimestamp > 0);

    m_tableComboBox->setEnabled(enableDataControls);
    m_plotButton->setEnabled(enableDataControls);

    m_lineWidthSpinBox->setEnabled(connected);
    m_titleSizeSpinBox->setEnabled(connected);
    m_titleBoldCheckBox->setEnabled(connected);
    m_labelSizeSpinBox->setEnabled(connected);
    m_labelBoldCheckBox->setEnabled(connected);
    m_trajectoryTitleEdit->setEnabled(connected);
    m_trajectoryXLabelEdit->setEnabled(connected);
    m_trajectoryYLabelEdit->setEnabled(connected);
    m_velocityTitleEdit->setEnabled(connected);
    m_velocityXLabelEdit->setEnabled(connected);
    m_velocityYLabelEdit->setEnabled(connected);
    m_errorTitleEdit->setEnabled(connected);
    m_errorXLabelEdit->setEnabled(connected);
    m_errorYLabelEdit->setEnabled(connected);

    if (m_liveUpdateCheckBox->isChecked() && enableDataControls && (m_maxTableTimestamp > 0)) {
        m_startTimeEdit->setEnabled(true);
        m_startRangeSlider->setEnabled(true);
        m_endTimeEdit->setEnabled(false);
        m_nowButton->setEnabled(false);
        m_endRangeSlider->setEnabled(false);
    } else {
        setTimeControlsEnabled(enableDataControls);
    }
    m_liveUpdateCheckBox->setEnabled(enableDataControls);
}

void MainWindow::showStatusMessage(const QString& message, int timeout) {
    m_statusBar->showMessage(message, timeout);
}

QCustomPlot* MainWindow::createPlotTab(const QString& title, QCPTextElement*& titleElementRef,
                                      QLineEdit* titleEditPtr, QLineEdit* xLabelEditPtr, QLineEdit* yLabelEditPtr) {
    QCustomPlot *customPlot = new QCustomPlot();
    m_tabWidget->addTab(customPlot, title);

    customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables | QCP::iSelectAxes);
    customPlot->axisRect()->setupFullAxesBox(true);

    // Corrected: Use addElement instead of insertElement for QCPLayoutGrid.
    titleElementRef = new QCPTextElement(customPlot);
    customPlot->plotLayout()->addElement(0, 0, titleElementRef); // Corrected this line

    return customPlot;
}

void MainWindow::onPlotStylesChanged() {
    updatePlotTextAndStyles(m_trajectoryPlot, m_trajectoryTitleElement,
                            m_trajectoryTitleEdit, m_trajectoryXLabelEdit, m_trajectoryYLabelEdit);
    updatePlotTextAndStyles(m_velocityPlot, m_velocityTitleElement,
                            m_velocityTitleEdit, m_velocityXLabelEdit, m_velocityYLabelEdit);
    updatePlotTextAndStyles(m_errorPlot, m_errorTitleElement,
                            m_errorTitleEdit, m_errorXLabelEdit, m_errorYLabelEdit);

    auto applyLineWidth = [&](QCustomPlot* plot) {
        if (plot) {
            for (int k = 0; k < plot->graphCount(); ++k) {
                QPen pen = plot->graph(k)->pen();
                pen.setWidth(m_lineWidthSpinBox->value());
                plot->graph(k)->setPen(pen);
            }
            plot->replot();
        }
    };

    applyLineWidth(m_trajectoryPlot);
    applyLineWidth(m_velocityPlot);
    applyLineWidth(m_errorPlot);
}

void MainWindow::updatePlotTextAndStyles(QCustomPlot* plot, QCPTextElement* titleElement,
                                    QLineEdit* titleEdit, QLineEdit* xLabelEdit, QLineEdit* yLabelEdit) {
    if (!plot || !titleElement) return;

    QFont titleFont = plot->font();
    titleFont.setPointSize(m_titleSizeSpinBox->value());
    titleFont.setBold(m_titleBoldCheckBox->isChecked());
    titleElement->setFont(titleFont);
    titleElement->setText(titleEdit->text());

    QFont labelFont = plot->font();
    labelFont.setPointSize(m_labelSizeSpinBox->value());
    labelFont.setBold(m_labelBoldCheckBox->isChecked());

    plot->xAxis->setLabelFont(labelFont);
    plot->yAxis->setLabelFont(labelFont);
    plot->xAxis->setTickLabelFont(labelFont);
    plot->yAxis->setTickLabelFont(labelFont);

    plot->xAxis->setLabel(xLabelEdit->text());
    plot->yAxis->setLabel(yLabelEdit->text());

    plot->replot();
}

void MainWindow::clearAllPlots() {
    if (m_trajectoryPlot) m_trajectoryPlot->clearGraphs();
    if (m_velocityPlot) m_velocityPlot->clearGraphs();
    if (m_errorPlot) m_errorPlot->clearGraphs();

    if (m_trajectoryPlot) m_trajectoryPlot->replot();
    if (m_velocityPlot) m_velocityPlot->replot();
    if (m_errorPlot) m_errorPlot->replot();
}

void MainWindow::plotAll() {
    clearAllPlots();

    if (m_robotDataCache.empty()) {
        showStatusMessage("没有数据可供绘图。", 3000);
        return;
    }

    plotTrajectories();
    plotVelocities();
    plotErrors();

    if (m_trajectoryPlot) {
        m_trajectoryPlot->rescaleAxes(true);
        // Corrected: Removed problematic line. For fixed aspect ratio, consider:
        // m_trajectoryPlot->axisRect()->setAspectRatio(1.0); // Set 1:1 ratio
        // m_trajectoryPlot->axisRect()->setMode(QCPAxisRect::amFixed); // Fix mode after setting ratio
    }
    if (m_velocityPlot) m_velocityPlot->rescaleAxes(true);
    if (m_errorPlot) m_errorPlot->rescaleAxes(true);

    onPlotStylesChanged();
    showStatusMessage("绘图完成.");
}

void MainWindow::plotTrajectories() {
    QVector<QColor> colors = {Qt::red, Qt::blue, Qt::green, Qt::darkYellow, Qt::magenta, Qt::cyan, Qt::gray, Qt::darkCyan, Qt::darkRed, Qt::darkGreen, Qt::darkBlue};
    int colorIndex = 0;

    auto getLTTBPointX = [](const LTTBPoint& p) { return p.x; };
    auto getLTTBPointY = [](const LTTBPoint& p) { return p.y; };

    for (auto const& [robotId, points] : m_robotDataCache) {
        if (points.isEmpty()) continue;

        qDebug() << "Plotting Robot" << robotId << "Trajectory. Original points:" << points.size();

        QVector<LTTBPoint> actualXYPoints;
        actualXYPoints.reserve(points.size());
        for (const auto& p : points) {
            actualXYPoints.append({p.actual_x, p.actual_y});
        }
        QVector<LTTBPoint> sampledActualXY = downsampleLTTB<LTTBPoint>(actualXYPoints, PLOT_DOWNSAMPLE_THRESHOLD,
                                                               getLTTBPointX, getLTTBPointY);
        QVector<double> actualX_sampled, actualY_sampled;
        actualX_sampled.reserve(sampledActualXY.size());
        actualY_sampled.reserve(sampledActualXY.size());
        for (const auto& p : sampledActualXY) {
            actualX_sampled.append(p.x);
            actualY_sampled.append(p.y);
        }
        qDebug() << "Sampled actual trajectory to" << actualX_sampled.size() << "points.";

        QVector<LTTBPoint> nashXYPoints;
        nashXYPoints.reserve(points.size());
        for (const auto& p : points) {
            nashXYPoints.append({p.nash_x, p.nash_y});
        }
        QVector<LTTBPoint> sampledNashXY = downsampleLTTB<LTTBPoint>(nashXYPoints, PLOT_DOWNSAMPLE_THRESHOLD,
                                                             getLTTBPointX, getLTTBPointY);
        QVector<double> nashX_sampled, nashY_sampled;
        nashX_sampled.reserve(sampledNashXY.size());
        nashY_sampled.reserve(sampledNashXY.size());
        for (const auto& p : sampledNashXY) {
            nashX_sampled.append(p.x);
            nashY_sampled.append(p.y);
        }
        qDebug() << "Sampled Nash trajectory to" << nashX_sampled.size() << "points.";

        QColor currentActualColor = colors[colorIndex % colors.size()];
        QColor currentNashColor = colors[colorIndex % colors.size()].lighter(150);

        m_trajectoryPlot->addGraph();
        m_trajectoryPlot->graph()->setData(actualX_sampled, actualY_sampled);
        m_trajectoryPlot->graph()->setName(QString("Robot %1 Actual").arg(robotId));
        m_trajectoryPlot->graph()->setPen(QPen(currentActualColor, m_lineWidthSpinBox->value()));
        m_trajectoryPlot->graph()->setLineStyle(QCPGraph::lsLine);
        m_trajectoryPlot->graph()->setScatterStyle(QCPScatterStyle::ssDisc);

        m_trajectoryPlot->addGraph();
        m_trajectoryPlot->graph()->setData(nashX_sampled, nashY_sampled);
        m_trajectoryPlot->graph()->setName(QString("Robot %1 Nash").arg(robotId));
        QPen nashPen(currentNashColor, m_lineWidthSpinBox->value(), Qt::DashDotLine);
        m_trajectoryPlot->graph()->setPen(nashPen);
        m_trajectoryPlot->graph()->setLineStyle(QCPGraph::lsLine);
        m_trajectoryPlot->graph()->setScatterStyle(QCPScatterStyle::ssCircle);

        colorIndex++;
    }
    m_trajectoryPlot->legend->setVisible(true);
    m_trajectoryPlot->legend->setFont(QFont("Helvetica", 9));
    m_trajectoryPlot->legend->setBrush(QBrush(QColor(255,255,255,200)));
}

void MainWindow::plotVelocities() {
    QVector<QColor> colors = {Qt::red, Qt::blue, Qt::green, Qt::darkYellow, Qt::magenta, Qt::cyan, Qt::gray, Qt::darkCyan, Qt::darkRed, Qt::darkGreen, Qt::darkBlue};
    int colorIndex = 0;

    auto getTimestamp = [](const PlotDataPoint& p) { return p.timestamp; };
    auto getLinearVel = [](const PlotDataPoint& p) { return p.linear_vel; };
    auto getAngularVel = [](const PlotDataPoint& p) { return p.angular_vel; };

    for (auto const& [robotId, points] : m_robotDataCache) {
        if (points.isEmpty()) continue;

        qDebug() << "Plotting Robot" << robotId << "Velocity. Original points:" << points.size();

        QVector<PlotDataPoint> sampledLinear = downsampleLTTB<PlotDataPoint>(points, PLOT_DOWNSAMPLE_THRESHOLD, getTimestamp, getLinearVel);
        QVector<double> timestamps_linear, linearVels;
        timestamps_linear.reserve(sampledLinear.size());
        linearVels.reserve(sampledLinear.size());
        for (const auto& p : sampledLinear) {
            timestamps_linear.append(p.timestamp - m_minTableTimestamp);
            linearVels.append(p.linear_vel);
        }
        qDebug() << "Sampled linear velocity to" << timestamps_linear.size() << "points.";

        QVector<PlotDataPoint> sampledAngular = downsampleLTTB<PlotDataPoint>(points, PLOT_DOWNSAMPLE_THRESHOLD, getTimestamp, getAngularVel);
        QVector<double> timestamps_angular, angularVels;
        timestamps_angular.reserve(sampledAngular.size());
        angularVels.reserve(sampledAngular.size());
        for (const auto& p : sampledAngular) {
            timestamps_angular.append(p.timestamp - m_minTableTimestamp);
            angularVels.append(p.angular_vel);
        }
        qDebug() << "Sampled angular velocity to" << timestamps_angular.size() << "points.";

        QColor currentLinearColor = colors[colorIndex % colors.size()];
        QColor currentAngularColor = colors[(colorIndex + 1) % colors.size()];

        m_velocityPlot->addGraph();
        m_velocityPlot->graph()->setData(timestamps_linear, linearVels);
        m_velocityPlot->graph()->setName(QString("Robot %1 Linear Vel").arg(robotId));
        m_velocityPlot->graph()->setPen(QPen(currentLinearColor, m_lineWidthSpinBox->value()));
        m_velocityPlot->graph()->setLineStyle(QCPGraph::lsLine);

        m_velocityPlot->addGraph();
        m_velocityPlot->graph()->setData(timestamps_angular, angularVels);
        m_velocityPlot->graph()->setName(QString("Robot %1 Angular Vel").arg(robotId));
        QPen angularPen(currentAngularColor, m_lineWidthSpinBox->value(), Qt::DashLine);
        m_velocityPlot->graph()->setPen(angularPen);
        m_velocityPlot->graph()->setLineStyle(QCPGraph::lsLine);

        colorIndex++;
    }
    m_velocityPlot->legend->setVisible(true);
}

void MainWindow::plotErrors() {
    QVector<QColor> colors = {Qt::red, Qt::blue, Qt::green, Qt::darkYellow, Qt::magenta, Qt::cyan, Qt::gray, Qt::darkCyan, Qt::darkRed, Qt::darkGreen, Qt::darkBlue};
    int colorIndex = 0;

    auto getTimestamp = [](const PlotDataPoint& p) { return p.timestamp; };
    auto getPositionError = [](const PlotDataPoint& p) {
        double error_x = p.actual_x - p.nash_x;
        double error_y = p.actual_y - p.nash_y;
        return std::sqrt(error_x * error_x + error_y * error_y);
    };

    for (auto const& [robotId, points] : m_robotDataCache) {
        if (points.isEmpty()) continue;

        qDebug() << "Plotting Robot" << robotId << "Errors. Original points:" << points.size();

        QVector<PlotDataPoint> sampledErrors = downsampleLTTB<PlotDataPoint>(points, PLOT_DOWNSAMPLE_THRESHOLD, getTimestamp, getPositionError);

        QVector<double> timestamps_error, errors;
        timestamps_error.reserve(sampledErrors.size());
        errors.reserve(sampledErrors.size());
        for (const auto& p : sampledErrors) {
            timestamps_error.append(p.timestamp - m_minTableTimestamp);
            double error_x = p.actual_x - p.nash_x;
            double error_y = p.actual_y - p.nash_y;
            errors.append(std::sqrt(error_x * error_x + error_y * error_y));
        }
        qDebug() << "Sampled errors to" << timestamps_error.size() << "points.";

        m_errorPlot->addGraph();
        m_errorPlot->graph()->setData(timestamps_error, errors);
        m_errorPlot->graph()->setName(QString("Robot %1 Position Error").arg(robotId));
        m_errorPlot->graph()->setPen(QPen(colors[colorIndex % colors.size()], m_lineWidthSpinBox->value()));
        m_errorPlot->graph()->setLineStyle(QCPGraph::lsLine);

        colorIndex++;
    }
    m_errorPlot->legend->setVisible(true);
}

void MainWindow::onStartRangeSliderValueChanged(int value) {
    if (m_maxTableTimestamp <= m_minTableTimestamp) return;

    double new_start_ts = m_minTableTimestamp + (m_maxTableTimestamp - m_minTableTimestamp) * (value / (double)SLIDER_RESOLUTION);
    QDateTime newStartTime = QDateTime::fromSecsSinceEpoch(static_cast<qint64>(new_start_ts));
    m_startTimeEdit->setDateTime(newStartTime);

    if (m_startTimeEdit->dateTime() > m_endTimeEdit->dateTime() && m_endRangeSlider->isEnabled()) {
        m_endRangeSlider->blockSignals(true);
        m_endTimeEdit->setDateTime(m_startTimeEdit->dateTime());
        int endSliderValue = qRound((m_endTimeEdit->dateTime().toSecsSinceEpoch() - m_minTableTimestamp) / (m_maxTableTimestamp - m_minTableTimestamp) * SLIDER_RESOLUTION);
        m_endRangeSlider->setValue(endSliderValue);
        m_endRangeSlider->blockSignals(false);
    }
}

void MainWindow::onEndRangeSliderValueChanged(int value) {
    if (m_maxTableTimestamp <= m_minTableTimestamp) return;

    double new_end_ts = m_minTableTimestamp + (m_maxTableTimestamp - m_minTableTimestamp) * (value / (double)SLIDER_RESOLUTION);
    QDateTime newEndTime = QDateTime::fromSecsSinceEpoch(static_cast<qint64>(new_end_ts));
    m_endTimeEdit->setDateTime(newEndTime);

    if (m_endTimeEdit->dateTime() < m_startTimeEdit->dateTime() && m_startRangeSlider->isEnabled()) {
        m_startRangeSlider->blockSignals(true);
        m_startTimeEdit->setDateTime(m_endTimeEdit->dateTime());
        int startSliderValue = qRound((m_startTimeEdit->dateTime().toSecsSinceEpoch() - m_minTableTimestamp) / (m_maxTableTimestamp - m_minTableTimestamp) * SLIDER_RESOLUTION);
        m_startRangeSlider->setValue(startSliderValue);
        m_startRangeSlider->blockSignals(false);
    }
}

void MainWindow::setTimeControlsEnabled(bool enabled) {
    bool hasValidTimeRange = (m_maxTableTimestamp > m_minTableTimestamp);

    m_startTimeEdit->setEnabled(enabled && hasValidTimeRange);
    m_endTimeEdit->setEnabled(enabled && hasValidTimeRange);
    m_nowButton->setEnabled(enabled && hasValidTimeRange);
    m_startRangeSlider->setEnabled(enabled && hasValidTimeRange);
    m_endRangeSlider->setEnabled(enabled && hasValidTimeRange);
    m_liveUpdateCheckBox->setEnabled(enabled && hasValidTimeRange);
    m_plotButton->setEnabled(enabled && hasValidTimeRange);

    if (m_liveUpdateCheckBox->isChecked() && enabled && hasValidTimeRange) {
        m_endTimeEdit->setDisabled(true);
        m_nowButton->setDisabled(true);
        m_endRangeSlider->setDisabled(true);
    }
}

void MainWindow::onLiveUpdateToggled(int state) {
    if (state == Qt::Checked) {
        m_updateTimer->start();
        m_endTimeEdit->setDisabled(true);
        m_nowButton->setDisabled(true);
        m_endRangeSlider->setDisabled(true);
    } else {
        m_updateTimer->stop();
        m_endTimeEdit->setDisabled(false);
        m_nowButton->setDisabled(false);
        m_endRangeSlider->setDisabled(false);
        fetchAndPlotData();
    }
}

// =========================================================================
// Main Entry Point
// =========================================================================

void runQt(int argc, char** argv) {
    qRegisterMetaType<PlotDataPoint>("PlotDataPoint");
    qRegisterMetaType<QVector<PlotDataPoint>>("QVector<PlotDataPoint>");
    qRegisterMetaType<RobotDataMap>("RobotDataMap");

    QApplication app(argc, argv);
    app.setStyle("Fusion");

    MainWindow mainWindow;
    mainWindow.show();

    app.exec();
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "mysql_read_node", ros::init_options::NoSigintHandler);

    std::thread qt_thread(runQt, argc, argv);

    ros::spin();

    if (qt_thread.joinable()) {
        qt_thread.join();
    }

    return 0;
}

// Ensure this matches your actual .cpp file name
#include "mysql_read.moc"
