import sys
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import lttb

# NEW: Plotly and WebEngine imports
import plotly.graph_objects as go
import plotly.colors as pcolors
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import json


from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QMessageBox,
    QTabWidget,
    QSpinBox,
    QDateTimeEdit,
    QSlider,
    QCheckBox,
    QGroupBox,
    QProgressBar,
    QStatusBar,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRunnable, QThreadPool, QDateTime, QTimer
from PyQt5.QtGui import QFont

import mysql.connector
from mysql.connector import Error as DB_Error


# =========================================================================
# 0. Plotly Widget Helper
# =========================================================================
class PlotlyWidget(QWebEngineView):
    """
    一个用于在PyQt中显示Plotly图表的QWidget。
    它加载一个包含Plotly.js的HTML模板，并提供一个方法来更新图表。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHtml(self._get_html_template(), QUrl("about:blank"))
        self.figure = go.Figure()

    def _get_html_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div id="plotdiv" style="width: 100%; height: 100vh;"></div>
            <script>
                function updatePlot(figure) {
                    Plotly.react('plotdiv', figure.data, figure.layout, {responsive: true});
                }
            </script>
        </body>
        </html>
        """

    def update_figure(self, fig: go.Figure):
        """
        使用新的figure对象更新图表。

        Args:
            fig: 一个 plotly.graph_objects.Figure 对象。
        """
        self.figure = fig
        fig_json = fig.to_json()
        self.page().runJavaScript(f"updatePlot({fig_json})")
    
    def clear(self):
        """清空图表"""
        self.update_figure(go.Figure())


# =========================================================================
# 1. Data Structures & Plotting Data Packages (Unchanged)
# =========================================================================
@dataclass
class PlotDataPoint:
    timestamp: float
    actual_x: float
    actual_y: float
    nash_x: float
    nash_y: float
    linear_vel: float
    angular_vel: float

@dataclass
class PlotLine:
    x_data: np.ndarray
    y_data: np.ndarray
    label: str
    # MODIFIED: Style attributes now friendlier to Plotly
    dash: str  # e.g., 'solid', 'dash', 'dot'
    color: Any
    mode: str = "lines+markers"
    marker_symbol: str = "circle"
    marker_size: int = 4

@dataclass
class ProcessedPlotData:
    success: bool
    message: str = ""
    point_count: int = 0
    trajectories: List[PlotLine] = field(default_factory=list)
    velocities: List[PlotLine] = field(default_factory=list)
    errors: List[PlotLine] = field(default_factory=list)

RobotDataMap = Dict[int, List[PlotDataPoint]]
DBParams = Tuple[str, str, str, str]


# =========================================================================
# 2. Worker/Manager Architecture (Modified for Plotly data)
# =========================================================================
class WorkerSignals(QObject):
    connection_result = pyqtSignal(bool, str)
    table_names_loaded = pyqtSignal(list)
    time_range_result = pyqtSignal(float, float, bool)
    plotting_data_ready = pyqtSignal(ProcessedPlotData)


class QueryAndProcessWorker(QRunnable):
    def __init__(self, db_manager, db_params: DBParams, signals: WorkerSignals, table_name: str, start_ts: float, end_ts: float):
        super().__init__()
        self.db_manager = db_manager
        self.db_params = db_params
        self.signals = signals
        self.table_name = table_name
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.downsample_threshold = MainWindow.PLOT_DOWNSAMPLE_THRESHOLD

    def _downsample(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(x) <= self.downsample_threshold:
            return x, y
        points = np.vstack((x, y)).T
        downsampled_points = lttb.downsample(points, n_out=self.downsample_threshold)
        return downsampled_points[:, 0], downsampled_points[:, 1]

    def run(self):
        try:
            raw_data, success, message = self.db_manager.query_data(
                self.db_params, self.table_name, self.start_ts, self.end_ts
            )
            if not success:
                self.signals.plotting_data_ready.emit(ProcessedPlotData(False, message))
                return
            if not raw_data:
                self.signals.plotting_data_ready.emit(ProcessedPlotData(True, "在指定时间范围内未找到数据"))
                return

            traj_lines, vel_lines, err_lines = [], [], []
            total_points = sum(len(v) for v in raw_data.values())
            num_robots = len(raw_data)
            
            # Use Plotly's color scales - they produce CSS compatible strings
            color_sequence = pcolors.qualitative.Plotly
            
            all_timestamps = [p.timestamp for pts in raw_data.values() for p in pts]
            start_time_offset = min(all_timestamps) if all_timestamps else 0

            for i, (rid, pts) in enumerate(raw_data.items()):
                if not pts:
                    continue

                color = color_sequence[i % len(color_sequence)]
                vel_color_1 = color_sequence[(i * 2) % len(color_sequence)]
                vel_color_2 = color_sequence[(i * 2 + 1) % len(color_sequence)]
                err_color = color

                ts = np.array([p.timestamp for p in pts]); ax = np.array([p.actual_x for p in pts])
                ay = np.array([p.actual_y for p in pts]); nx = np.array([p.nash_x for p in pts])
                ny = np.array([p.nash_y for p in pts]); lv = np.array([p.linear_vel for p in pts])
                av = np.array([p.angular_vel for p in pts]); pe = np.sqrt((ax - nx) ** 2 + (ay - ny) ** 2)

                ax_down, ay_down = self._downsample(ax, ay)
                traj_lines.append(PlotLine(ax_down, ay_down, f"R{rid} Actual", "solid", color))
                nx_down, ny_down = self._downsample(nx, ny)
                traj_lines.append(PlotLine(nx_down, ny_down, f"R{rid} Nash", "dash", color, marker_symbol="circle-open", marker_size=6))

                ts_rel = ts - start_time_offset
                ts_lv_down, lv_down = self._downsample(ts_rel, lv)
                vel_lines.append(PlotLine(ts_lv_down, lv_down, f"R{rid} Linear Vel", "solid", vel_color_1))
                ts_av_down, av_down = self._downsample(ts_rel, av)
                vel_lines.append(PlotLine(ts_av_down, av_down, f"R{rid} Angular Vel", "dash", vel_color_2))

                ts_pe_down, pe_down = self._downsample(ts_rel, pe)
                err_lines.append(PlotLine(ts_pe_down, pe_down, f"R{rid} Position Err", "solid", err_color))

            self.signals.plotting_data_ready.emit(
                ProcessedPlotData(True, trajectories=traj_lines, velocities=vel_lines, errors=err_lines, point_count=total_points)
            )
        except AttributeError as e:
            if "lttb" in str(e):
                self.signals.plotting_data_ready.emit(ProcessedPlotData(False, f"`lttb`库函数调用错误: {e}. 请确保安装了正确的`lttb`包。"))
            else:
                self.signals.plotting_data_ready.emit(ProcessedPlotData(False, f"处理数据时出错: {e}"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.plotting_data_ready.emit(ProcessedPlotData(False, f"处理数据时出错: {e}"))

# GenericWorker and DatabaseManager remain unchanged from the original code
class GenericWorker(QRunnable):
    def __init__(self, fn, signals: WorkerSignals, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.signals = signals
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            fn_name = self.fn.__name__
            if fn_name == "test_connection":
                self.signals.connection_result.emit(*res)
            elif fn_name == "load_table_names":
                self.signals.table_names_loaded.emit(res)
            elif fn_name == "get_time_range":
                self.signals.time_range_result.emit(*res)
        except Exception as e:
            print(f"工作线程错误: {e}")
            if self.fn.__name__ == "test_connection":
                self.signals.connection_result.emit(False, str(e))


class DatabaseManager:
    def _get_connection_dict(self, db_params: DBParams) -> Dict[str, Any]:
        h, u, p, d = db_params
        hp_parts = h.replace("tcp://", "").split(":")
        host_addr = hp_parts[0]
        port = int(hp_parts[1]) if len(hp_parts) > 1 else 3306
        return {
            "host": host_addr, "port": port, "user": u, "password": p, "database": d, "connect_timeout": 10
        }

    def test_connection(self, h: str, u: str, p: str, d: str) -> Tuple[bool, str]:
        conn = None
        try:
            conn = mysql.connector.connect(**self._get_connection_dict((h, u, p, d)))
            return (True, "连接成功") if conn.is_connected() else (False, "连接失败")
        except DB_Error as e:
            return False, f"连接失败: {e.msg} (错误码: {e.errno})"
        except Exception as e:
            return False, f"连接失败: {e}"
        finally:
            if conn and conn.is_connected():
                conn.close()

    def load_table_names(self, db_params: DBParams) -> List[str]:
        conn = None
        try:
            conn = mysql.connector.connect(**self._get_connection_dict(db_params))
            with conn.cursor() as c:
                c.execute("SHOW TABLES")
                return [""] + [t[0] for t in c.fetchall()]
        except DB_Error as e:
            print(f"加载表失败: {e}")
            return [""]
        finally:
            if conn and conn.is_connected():
                conn.close()

    def get_time_range(self, db_params: DBParams, table_name: str) -> Tuple[float, float, bool]:
        if not table_name:
            return 0.0, 0.0, False
        conn = None
        try:
            conn = mysql.connector.connect(**self._get_connection_dict(db_params))
            with conn.cursor() as c:
                c.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM `{table_name}` WHERE timestamp IS NOT NULL")
                result = c.fetchone()
                if result and result[0] is not None and result[1] is not None:
                    return float(result[0]), float(result[1]), True
                return 0.0, 0.0, False
        except DB_Error as e:
            print(f"获取时间范围失败: {e}")
            return 0.0, 0.0, False
        finally:
            if conn and conn.is_connected():
                conn.close()

    def query_data(self, db_params: DBParams, table_name: str, start_ts: float, end_ts: float) -> Tuple[RobotDataMap, bool, str]:
        if not table_name:
            return {}, False, "未选择数据表"
        conn = None
        try:
            conn = mysql.connector.connect(**self._get_connection_dict(db_params))
            cache: RobotDataMap = {}
            with conn.cursor(dictionary=True) as c:
                query = (
                    f"SELECT timestamp, robot_id, actual_x, actual_y, nash_x, nash_y, linear_vel, angular_vel "
                    f"FROM `{table_name}` WHERE timestamp BETWEEN %s AND %s ORDER BY robot_id, timestamp ASC"
                )
                c.execute(query, (start_ts, end_ts))
                for row in c.fetchall():
                    rid = int(row["robot_id"])
                    if rid not in cache:
                        cache[rid] = []
                    cache[rid].append(
                        PlotDataPoint(
                            timestamp=row.get("timestamp", 0.0),
                            actual_x=row.get("actual_x", 0.0),
                            actual_y=row.get("actual_y", 0.0),
                            nash_x=row.get("nash_x", 0.0),
                            nash_y=row.get("nash_y", 0.0),
                            linear_vel=row.get("linear_vel", 0.0),
                            angular_vel=row.get("angular_vel", 0.0),
                        )
                    )
            return cache, True, "查询成功"
        except DB_Error as e:
            print(f"查询失败: {e}")
            return {}, False, f"查询失败: {e}"
        finally:
            if conn and conn.is_connected():
                conn.close()

# =========================================================================
# 3. MainWindow (Rewritten for Plotly)
# =========================================================================
class MainWindow(QMainWindow):
    SLIDER_RESOLUTION = 10000
    PLOT_DOWNSAMPLE_THRESHOLD = 4000
    LIVE_UPDATE_INTERVAL = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread_pool = QThreadPool()
        self.db_manager = DatabaseManager()
        self.db_params: Optional[DBParams] = None
        self.is_loading = False
        self.min_table_timestamp = 0.0
        self.max_table_timestamp = 0.0
        self.is_connected = False
        self.signals = WorkerSignals()
        
        # Structure to hold plot objects
        self.plots = {}

        self.setupUi()
        self.connect_signals()
        QTimer.singleShot(100, self.update_ui_state)

    def setupUi(self):
        self.setWindowTitle("高性能MySQL数据可视化 (Plotly版)")
        self.resize(1800, 1000)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.setStatusBar(QStatusBar(self))
        status_bar = self.statusBar()
        self.progress_bar = QProgressBar(status_bar)
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        status_bar.addPermanentWidget(self.progress_bar)
        top_panel = QWidget()
        top_layout = QGridLayout(top_panel)
        main_layout.addWidget(top_panel)

        self.tab_widget = QTabWidget()
        self._create_plot_tab("trajectory", "轨迹图", "世界坐标 X (m)", "世界坐标 Y (m)")
        self._create_plot_tab("velocity", "速度图", "时间 (s)", "速度 (m/s, rad/s)")
        self._create_plot_tab("error", "误差图", "时间 (s)", "位置误差 (m)")
  
        self._setup_connection_widgets(top_layout)
        self._setup_controls(top_layout)

        main_layout.addWidget(self.tab_widget)
        
    def _create_plot_tab(self, name: str, title: str, xlabel: str, ylabel: str):
        container_widget = QWidget()
        layout = QVBoxLayout(container_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # Axis range controls
        range_widget = QWidget()
        range_layout = QHBoxLayout(range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        xmin_edit = QLineEdit("-10")
        xmax_edit = QLineEdit("10")
        ymin_edit = QLineEdit("-10")
        ymax_edit = QLineEdit("10")
        range_layout.addWidget(QLabel("X范围:"))
        range_layout.addWidget(xmin_edit); range_layout.addWidget(QLabel("到")); range_layout.addWidget(xmax_edit)
        range_layout.addSpacing(20)
        range_layout.addWidget(QLabel("Y范围:"))
        range_layout.addWidget(ymin_edit); range_layout.addWidget(QLabel("到")); range_layout.addWidget(ymax_edit)
        range_layout.addStretch()
        layout.addWidget(range_widget)
        
        # Plotly widget
        plot_widget = PlotlyWidget()
        layout.addWidget(plot_widget)

        # Store all components in a dictionary for easy access
        self.plots[name] = {
            "fig": go.Figure(),
            "widget": plot_widget,
            "title_edit": QLineEdit(title),
            "xlabel_edit": QLineEdit(xlabel),
            "ylabel_edit": QLineEdit(ylabel),
            "xmin_edit": xmin_edit,
            "xmax_edit": xmax_edit,
            "ymin_edit": ymin_edit,
            "ymax_edit": ymax_edit,
        }
        self.tab_widget.addTab(container_widget, title)

    def _setup_plot_settings_controls(self, layout: QGridLayout):
        self.plot_settings_group = QGroupBox("绘图设置")
        psl = QGridLayout(self.plot_settings_group)
  
        self.title_size_spinbox = QSpinBox(); self.title_size_spinbox.setRange(8, 30); self.title_size_spinbox.setValue(16)
        self.label_size_spinbox = QSpinBox(); self.label_size_spinbox.setRange(8, 24); self.label_size_spinbox.setValue(12)
        self.line_width_spinbox = QSpinBox(); self.line_width_spinbox.setRange(1, 10); self.line_width_spinbox.setValue(2)
        self.legend_size_spinbox = QSpinBox(); self.legend_size_spinbox.setRange(8, 20); self.legend_size_spinbox.setValue(11)

        psl.addWidget(QLabel("标题字号:"), 0, 0); psl.addWidget(self.title_size_spinbox, 0, 1)        
        psl.addWidget(QLabel("标签字号:"), 0, 2); psl.addWidget(self.label_size_spinbox, 0, 3)
        psl.addWidget(QLabel("图例字号:"), 0, 4); psl.addWidget(self.legend_size_spinbox, 0, 5)
        psl.addWidget(QLabel("线宽:"), 0, 6); psl.addWidget(self.line_width_spinbox, 0, 7)
        
        def add_plot_specific_settings(name, row):
            plot_widgets = self.plots[name]
            psl.addWidget(QLabel(f"{name.capitalize()} 标题:"), row, 0)
            psl.addWidget(plot_widgets["title_edit"], row, 1, 1, 2)
            psl.addWidget(QLabel("X轴:"), row, 3); psl.addWidget(plot_widgets["xlabel_edit"], row, 4)
            psl.addWidget(QLabel("Y轴:"), row, 5); psl.addWidget(plot_widgets["ylabel_edit"], row, 6)

        add_plot_specific_settings("trajectory", 1)
        add_plot_specific_settings("velocity", 2)
        add_plot_specific_settings("error", 3)
  
        self.apply_ranges_button = QPushButton("应用范围")
        self.auto_range_button = QPushButton("自动范围")
        self.redraw_button = QPushButton("重绘样式")
        psl.addWidget(self.apply_ranges_button, 4, 1)
        psl.addWidget(self.auto_range_button, 4, 2)
        psl.addWidget(self.redraw_button, 4, 3)

        layout.addWidget(self.plot_settings_group, 4, 0, 1, 10)

    # _setup_connection_widgets and _setup_controls remain largely the same
    def _setup_connection_widgets(self, layout: QGridLayout):
        layout.addWidget(QLabel("主机:"), 0, 0); self.host_line_edit = QLineEdit("tcp://127.0.0.1:3306"); layout.addWidget(self.host_line_edit, 0, 1, 1, 2)
        layout.addWidget(QLabel("用户:"), 0, 3); self.user_line_edit = QLineEdit("car_user"); layout.addWidget(self.user_line_edit, 0, 4)
        layout.addWidget(QLabel("密码:"), 0, 5); self.pass_line_edit = QLineEdit("StrongPassword123!"); self.pass_line_edit.setEchoMode(QLineEdit.Password); layout.addWidget(self.pass_line_edit, 0, 6)
        layout.addWidget(QLabel("数据库:"), 0, 7); self.db_line_edit = QLineEdit("car_tracking"); layout.addWidget(self.db_line_edit, 0, 8)
        self.connect_button = QPushButton("连接"); layout.addWidget(self.connect_button, 0, 9)

    def _setup_controls(self, layout: QGridLayout):
        row = 1
        layout.addWidget(QLabel("数据表:"), row, 0); self.table_combo_box = QComboBox(); layout.addWidget(self.table_combo_box, row, 1, 1, 2)
        layout.addWidget(QLabel("开始:"), row, 3); self.start_time_edit = QDateTimeEdit(); self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss"); self.start_time_edit.setCalendarPopup(True); layout.addWidget(self.start_time_edit, row, 4, 1, 2)
        layout.addWidget(QLabel("结束:"), row, 6); self.end_time_edit = QDateTimeEdit(QDateTime.currentDateTime()); self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss"); self.end_time_edit.setCalendarPopup(True); layout.addWidget(self.end_time_edit, row, 7, 1, 2)
        self.now_button = QPushButton("现在"); layout.addWidget(self.now_button, row, 9)

        row = 2
        self.live_update_checkbox = QCheckBox("实时更新"); layout.addWidget(self.live_update_checkbox, row, 0)
        self.start_range_slider = QSlider(Qt.Horizontal); self.start_range_slider.setRange(0, self.SLIDER_RESOLUTION); layout.addWidget(self.start_range_slider, row, 1, 1, 9)

        row = 3
        self.plot_button = QPushButton("绘图"); self.plot_button.setStyleSheet("QPushButton{background-color:#4CAF50;color:white;border-radius:5px;padding:6px}QPushButton:hover{background-color:#45a049}QPushButton:disabled{background-color:#cccccc}"); layout.addWidget(self.plot_button, row, 0)
        self.end_range_slider = QSlider(Qt.Horizontal); self.end_range_slider.setRange(0, self.SLIDER_RESOLUTION); self.end_range_slider.setValue(self.SLIDER_RESOLUTION); layout.addWidget(self.end_range_slider, row, 1, 1, 9)
  
        self._setup_plot_settings_controls(layout)

    def connect_signals(self):
        self.signals.connection_result.connect(self.handle_connection_result)
        self.signals.table_names_loaded.connect(self.handle_table_names_loaded)
        self.signals.time_range_result.connect(self.handle_time_range_result)
        self.signals.plotting_data_ready.connect(self.handle_plotting_data_ready)

        self.connect_button.clicked.connect(self.run_db_connect)
        self.plot_button.clicked.connect(self.fetch_and_plot_data)
        self.table_combo_box.currentTextChanged.connect(self.on_table_name_changed)
        self.live_update_checkbox.stateChanged.connect(self.on_live_update_toggled)
        self.now_button.clicked.connect(lambda: self.end_time_edit.setDateTime(QDateTime.currentDateTime()))
        self.start_range_slider.valueChanged.connect(self.on_start_slider_changed)
        self.end_range_slider.valueChanged.connect(self.on_end_slider_changed)

        self.redraw_button.clicked.connect(self.update_all_plot_styles)
        self.apply_ranges_button.clicked.connect(self.on_apply_ranges)
        self.auto_range_button.clicked.connect(self.on_auto_range)

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(self.LIVE_UPDATE_INTERVAL)
        self.update_timer.timeout.connect(self.fetch_and_plot_data)

    def plot_data(self, data: ProcessedPlotData):
        """Main function to update all plots with new data."""
        self._update_figure_from_data("trajectory", data.trajectories)
        self._update_figure_from_data("velocity", data.velocities)
        self._update_figure_from_data("error", data.errors)
        self.update_all_plot_styles()

    def _update_figure_from_data(self, name: str, lines: List[PlotLine]):
        """Generates a new figure with traces from PlotLine data."""
        plot_info = self.plots[name]
        fig = go.Figure() # Create a fresh figure
        
        line_width = self.line_width_spinbox.value()
        
        for line in lines:
            fig.add_trace(go.Scatter(
                x=line.x_data,
                y=line.y_data,
                name=line.label,
                mode=line.mode,
                line=dict(color=line.color, width=line_width, dash=line.dash),
                marker=dict(symbol=line.marker_symbol, size=line.marker_size)
            ))
        
        if name == "trajectory":
             # Keep aspect ratio for trajectory plot
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        plot_info["fig"] = fig # Store the new figure with data
    
    def update_all_plot_styles(self):
        """Applies current style settings to all plots and redraws them."""
        for name, plot_info in self.plots.items():
            fig = plot_info["fig"]
            widget = plot_info["widget"]
            
            # Apply layout styles from UI controls
            fig.update_layout(
                title_text=plot_info["title_edit"].text(),
                xaxis_title=plot_info["xlabel_edit"].text(),
                yaxis_title=plot_info["ylabel_edit"].text(),
                title_font_size=self.title_size_spinbox.value(),
                xaxis_title_font_size=self.label_size_spinbox.value(),
                yaxis_title_font_size=self.label_size_spinbox.value(),
                legend_font_size=self.legend_size_spinbox.value(),
                legend=dict(x=1, y=1, xanchor='right', yanchor='top', bgcolor='rgba(255,255,255,0.5)'),
                margin=dict(l=50, r=20, t=50, b=40),
            )
            widget.update_figure(fig)

    def on_apply_ranges(self):
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index == -1: return
        plot_name = list(self.plots.keys())[current_tab_index]
        plot_info = self.plots[plot_name]

        try:
            xmin = float(plot_info["xmin_edit"].text())
            xmax = float(plot_info["xmax_edit"].text())
            ymin = float(plot_info["ymin_edit"].text())
            ymax = float(plot_info["ymax_edit"].text())
            
            if xmin >= xmax or ymin >= ymax:
                QMessageBox.warning(self, "范围错误", "最小值必须小于最大值。")
                return
            
            fig = plot_info["fig"]
            fig.update_xaxes(range=[xmin, xmax])
            fig.update_yaxes(range=[ymin, ymax])
            plot_info["widget"].update_figure(fig)
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请输入有效的数字作为坐标轴范围。")

    def on_auto_range(self):
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index == -1: return
        plot_name = list(self.plots.keys())[current_tab_index]
        plot_info = self.plots[plot_name]
        fig = plot_info["fig"]
        
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)
        plot_info["widget"].update_figure(fig)
    
    def handle_plotting_data_ready(self, data: ProcessedPlotData):
        self.clear_all_plots()
        if data.success:
            self.plot_data(data)
            msg = f"绘图完成。处理点数: {data.point_count}" + (f" | {data.message}" if data.message else "")
            self.show_status_message(msg, 5000)
        else:
            msg = data.message or "绘图失败"
            self.show_status_message(msg, 5000)
            QMessageBox.warning(self, "绘图信息", msg)
        self.set_loading_state(False)

    def clear_all_plots(self):
        for name, plot_info in self.plots.items():
            plot_info["widget"].clear()
            plot_info["fig"] = go.Figure() # Reset the figure object

    # All other methods (UI state, workers, signals, etc.) are mostly unchanged
    # Minor adjustments might be needed to call the new functions
    # For brevity, I'll only show modified methods
    def on_table_name_changed(self, table_name: str):
        self.clear_all_plots()
        self.min_table_timestamp, self.max_table_timestamp = 0.0, 0.0
        if table_name and self.db_params:
            self.set_loading_state(True, f"获取 '{table_name}' 时间范围...")
            worker = GenericWorker(self.db_manager.get_time_range, self.signals, self.db_params, table_name)
            self.thread_pool.start(worker)
        else:
            self.update_ui_state()

    def update_ui_state(self):
        has_time_range = self.min_table_timestamp > 0 and self.max_table_timestamp > 0
        is_live_update_enabled = self.live_update_checkbox.isChecked()
        can_interact = not self.is_loading
        self.connect_button.setEnabled(not self.is_connected and can_interact)
        for w in [self.host_line_edit, self.user_line_edit, self.pass_line_edit, self.db_line_edit]:
            w.setEnabled(not self.is_connected and can_interact)
        self.plot_settings_group.setEnabled(self.is_connected and can_interact)
        self.table_combo_box.setEnabled(self.is_connected and can_interact)
        can_plot = self.is_connected and has_time_range and can_interact
        self.plot_button.setEnabled(can_plot and not is_live_update_enabled)
        self.live_update_checkbox.setEnabled(can_plot)
        self.start_time_edit.setEnabled(can_plot)
        self.start_range_slider.setEnabled(can_plot)
        self.end_time_edit.setEnabled(can_plot and not is_live_update_enabled)
        self.end_range_slider.setEnabled(can_plot and not is_live_update_enabled)
        self.now_button.setEnabled(can_plot and not is_live_update_enabled)

    def set_loading_state(self, loading: bool, message: str = ""):
        self.is_loading = loading
        self.progress_bar.setVisible(loading)
        self.progress_bar.setRange(0, 0 if loading else 1)
        if message:
            self.show_status_message(message)
        self.update_ui_state()
        QApplication.processEvents()

    def run_db_connect(self):
        self.set_loading_state(True, "连接中...")
        h = self.host_line_edit.text()
        u = self.user_line_edit.text()
        p = self.pass_line_edit.text()
        d = self.db_line_edit.text()
        if not all([h, u, d]):
            QMessageBox.warning(self, "输入错误", "请填写主机、用户和数据库字段。")
            self.set_loading_state(False)
            return
        worker = GenericWorker(self.db_manager.test_connection, self.signals, h, u, p, d)
        self.thread_pool.start(worker)

    def fetch_and_plot_data(self):
        if self.is_loading: return
        if not self.db_params:
            if not self.live_update_checkbox.isChecked(): QMessageBox.warning(self, "错误", "未连接到数据库。")
            return
        if self.live_update_checkbox.isChecked(): self.end_time_edit.setDateTime(QDateTime.currentDateTime())
        table_name = self.table_combo_box.currentText()
        if not table_name:
            if not self.live_update_checkbox.isChecked(): QMessageBox.warning(self, "提示", "请选择一张数据表。")
            return
        start_ts = self.start_time_edit.dateTime().toSecsSinceEpoch()
        end_ts = self.end_time_edit.dateTime().toSecsSinceEpoch()
        if start_ts > end_ts:
            if not self.live_update_checkbox.isChecked(): QMessageBox.warning(self, "错误", "开始时间不能晚于结束时间。")
            return
        self.set_loading_state(True, "查询和处理数据中...")
        worker = QueryAndProcessWorker(self.db_manager, self.db_params, self.signals, table_name, start_ts, end_ts)
        self.thread_pool.start(worker)
  
    def handle_connection_result(self, success: bool, message: str):
        if success:
            self.is_connected = True
            self.db_params = (self.host_line_edit.text(), self.user_line_edit.text(), self.pass_line_edit.text(), self.db_line_edit.text())
            QMessageBox.information(self, "成功", message)
            self.set_loading_state(True, "加载数据表...")
            worker = GenericWorker(self.db_manager.load_table_names, self.signals, self.db_params)
            self.thread_pool.start(worker)
        else:
            self.is_connected = False
            self.db_params = None
            QMessageBox.critical(self, "错误", message)
            self.set_loading_state(False, "连接失败")

    def handle_table_names_loaded(self, tables: List[str]):
        self.table_combo_box.clear()
        self.table_combo_box.addItems(tables)
        if len(tables) > 1: self.table_combo_box.setCurrentIndex(1)
        self.set_loading_state(False, "表加载完成")

    def handle_time_range_result(self, min_ts: float, max_ts: float, success: bool):
        self.min_table_timestamp = 0.0
        self.max_table_timestamp = 0.0
        if success:
            self.min_table_timestamp, self.max_table_timestamp = min_ts, max_ts
            self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(min_ts)))
            self.end_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(max_ts)))
            self.start_range_slider.setValue(0)
            self.end_range_slider.setValue(self.SLIDER_RESOLUTION)
            self.set_loading_state(False, "时间范围已获取")
            self.fetch_and_plot_data()
        else:
            self.set_loading_state(False, "无法获取时间范围")
            self.clear_all_plots()

    def on_live_update_toggled(self, state: int):
        self.update_ui_state()
        if state == Qt.Checked:
            self.update_timer.start()
            self.end_time_edit.setDateTime(QDateTime.currentDateTime())
            self.fetch_and_plot_data()
        else:
            self.update_timer.stop()

    def on_start_slider_changed(self, value: int):
        if self.max_table_timestamp <= self.min_table_timestamp: return
        ts = self.min_table_timestamp + (self.max_table_timestamp - self.min_table_timestamp) * (value / self.SLIDER_RESOLUTION)
        self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(ts)))
        if value > self.end_range_slider.value(): self.end_range_slider.setValue(value)

    def on_end_slider_changed(self, value: int):
        if self.max_table_timestamp <= self.min_table_timestamp: return
        ts = self.min_table_timestamp + (self.max_table_timestamp - self.min_table_timestamp) * (value / self.SLIDER_RESOLUTION)
        self.end_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(ts)))
        if value < self.start_range_slider.value(): self.start_range_slider.setValue(value)

    def show_status_message(self, message: str, timeout: int = 0):
        self.statusBar().showMessage(message, timeout)

    def closeEvent(self, event):
        self.thread_pool.waitForDone(-1)
        event.accept()

if __name__ == "__main__":
    # 确保在创建 QApplication 实例之前设置此属性，以允许跨域加载JS库
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


# import sys
# import math
# from datetime import datetime
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Tuple, Any

# import numpy as np
# import lttb

# from PyQt5.QtWidgets import (
#     QApplication,
#     QMainWindow,
#     QWidget,
#     QVBoxLayout,
#     QGridLayout,
#     QLabel,
#     QLineEdit,
#     QPushButton,
#     QComboBox,
#     QMessageBox,
#     QTabWidget,
#     QSpinBox,
#     QDateTimeEdit,
#     QSlider,
#     QCheckBox,
#     QGroupBox,
#     QProgressBar,
#     QStatusBar,
#     QHBoxLayout,
# )
# from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRunnable, QThreadPool, QDateTime, QTimer
# from PyQt5.QtGui import QFont

# import matplotlib

# matplotlib.use("Qt5Agg")
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
# from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# import mysql.connector
# from mysql.connector import Error as DB_Error


# # =========================================================================
# # 0. Utility Functions
# # =========================================================================
# def find_chinese_font():
#     """在系统中查找可用的中文字体并设置"""
#     font_names = ["WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "Arial Unicode MS", "Heiti TC", "sans-serif"]
#     for font_name in font_names:
#         try:
#             if fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False):
#                 plt.rcParams["font.sans-serif"] = [font_name]
#                 plt.rcParams["axes.unicode_minus"] = False
#                 print(f"成功加载中文字体: {font_name}")
#                 return True
#         except ValueError:
#             continue

#     print("\n" + "=" * 50)
#     print("【警告】未找到任何预设的中文字体。中文将显示为方框。")
#     print("请按以下步骤安装字体后重试：")
#     print("1. Debian/Ubuntu: sudo apt-get install -y fonts-wqy-zenhei")
#     print("2. CentOS/Fedora: sudo yum install -y wqy-zenhei-fonts")
#     print("3. 安装后，删除Matplotlib缓存: rm -rf ~/.matplotlib")
#     print("程序将继续运行，但标签和标题将使用默认英文字体。")
#     print("=" * 50 + "\n")
#     return False


# # =========================================================================
# # 1. Data Structures & Plotting Data Packages
# # =========================================================================
# @dataclass
# class PlotDataPoint:
#     timestamp: float
#     actual_x: float
#     actual_y: float
#     nash_x: float
#     nash_y: float
#     linear_vel: float
#     angular_vel: float


# @dataclass
# class PlotLine:
#     x_data: np.ndarray
#     y_data: np.ndarray
#     label: str
#     style: str
#     color: Any
#     marker: str = "."
#     markersize: int = 2
#     fillstyle: str = "full"


# @dataclass
# class ProcessedPlotData:
#     success: bool
#     message: str = ""
#     point_count: int = 0
#     trajectories: List[PlotLine] = field(default_factory=list)
#     velocities: List[PlotLine] = field(default_factory=list)
#     errors: List[PlotLine] = field(default_factory=list)


# RobotDataMap = Dict[int, List[PlotDataPoint]]
# DBParams = Tuple[str, str, str, str]


# # =========================================================================
# # [NEW] Draggable Pan and Zoom Helper Class
# # =========================================================================
# class DraggablePanAndZoom:
#     """一个封装了拖动平移和滚轮缩放逻辑的类"""

#     def __init__(self, fig: Figure, ax: plt.Axes):
#         self.figure = fig
#         self.ax = ax
#         self.press = None
#         self.x0, self.y0 = 0, 0
#         self.xlim, self.ylim = ax.get_xlim(), ax.get_ylim()
#         self.connect()

#     def connect(self):
#         self.cid_press = self.figure.canvas.mpl_connect("button_press_event", self.on_press)
#         self.cid_release = self.figure.canvas.mpl_connect("button_release_event", self.on_release)
#         self.cid_motion = self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
#         self.cid_scroll = self.figure.canvas.mpl_connect("scroll_event", self.on_scroll)

#     def on_press(self, event):
#         if event.inaxes != self.ax or event.button != 1:
#             return  # 只响应左键
#         self.press = True
#         self.x0, self.y0 = event.xdata, event.ydata
#         self.xlim, self.ylim = self.ax.get_xlim(), self.ax.get_ylim()
#         QApplication.setOverrideCursor(Qt.OpenHandCursor)

#     def on_motion(self, event):
#         if not self.press or event.inaxes != self.ax:
#             return
#         if event.xdata is None or event.ydata is None:
#             return
#         dx = event.xdata - self.x0
#         dy = event.ydata - self.y0
#         new_xlim = (self.xlim[0] - dx, self.xlim[1] - dx)
#         new_ylim = (self.ylim[0] - dy, self.ylim[1] - dy)
#         self.ax.set_xlim(new_xlim)
#         self.ax.set_ylim(new_ylim)
#         self.figure.canvas.draw_idle()

#     def on_release(self, event):
#         self.press = False
#         QApplication.restoreOverrideCursor()

#     def on_scroll(self, event):
#         if event.inaxes != self.ax:
#             return
#         base_scale = 1.1
#         if event.button == "up":
#             scale_factor = 1 / base_scale  # 放大
#         elif event.button == "down":
#             scale_factor = base_scale  # 缩小
#         else:
#             return

#         cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
#         xdata, ydata = event.xdata, event.ydata

#         if xdata is None or ydata is None: return # 避免鼠标在图外滚动

#         new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
#         new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

#         rel_x = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
#         rel_y = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

#         self.ax.set_xlim([xdata - new_width * rel_x, xdata + new_width * (1 - rel_x)])
#         self.ax.set_ylim([ydata - new_height * rel_y, ydata + new_height * (1 - rel_y)])
#         self.figure.canvas.draw_idle()

#     def disconnect(self):
#         self.figure.canvas.mpl_disconnect(self.cid_press)
#         self.figure.canvas.mpl_disconnect(self.cid_release)
#         self.figure.canvas.mpl_disconnect(self.cid_motion)
#         self.figure.canvas.mpl_disconnect(self.cid_scroll)


# # =========================================================================
# # 2. Worker/Manager Architecture
# # =========================================================================
# class WorkerSignals(QObject):
#     connection_result = pyqtSignal(bool, str)
#     table_names_loaded = pyqtSignal(list)
#     time_range_result = pyqtSignal(float, float, bool)
#     plotting_data_ready = pyqtSignal(ProcessedPlotData)


# class QueryAndProcessWorker(QRunnable):
#     def __init__(self, db_manager, db_params: DBParams, signals: WorkerSignals, table_name: str, start_ts: float, end_ts: float):
#         super().__init__()
#         self.db_manager = db_manager
#         self.db_params = db_params
#         self.signals = signals
#         self.table_name = table_name
#         self.start_ts = start_ts
#         self.end_ts = end_ts
#         self.downsample_threshold = MainWindow.PLOT_DOWNSAMPLE_THRESHOLD

#     def _downsample(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         if len(x) <= self.downsample_threshold:
#             return x, y
#         points = np.vstack((x, y)).T
#         downsampled_points = lttb.downsample(points, n_out=self.downsample_threshold)
#         return downsampled_points[:, 0], downsampled_points[:, 1]

#     def run(self):
#         try:
#             raw_data, success, message = self.db_manager.query_data(
#                 self.db_params, self.table_name, self.start_ts, self.end_ts
#             )
#             if not success:
#                 self.signals.plotting_data_ready.emit(ProcessedPlotData(False, message))
#                 return
#             if not raw_data:
#                 self.signals.plotting_data_ready.emit(ProcessedPlotData(True, "在指定时间范围内未找到数据"))
#                 return

#             traj_lines, vel_lines, err_lines = [], [], []
#             total_points = sum(len(v) for v in raw_data.values())
#             num_robots = len(raw_data)
#             traj_colors = plt.cm.viridis(np.linspace(0, 1, max(1, num_robots)))
#             vel_colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(1, num_robots * 2)))
#             err_colors = plt.cm.inferno(np.linspace(0.1, 0.9, max(1, num_robots)))
#             all_timestamps = [p.timestamp for pts in raw_data.values() for p in pts]
#             start_time_offset = min(all_timestamps) if all_timestamps else 0

#             for i, (rid, pts) in enumerate(raw_data.items()):
#                 if not pts:
#                     continue
                
#                 ts = np.array([p.timestamp for p in pts]); ax = np.array([p.actual_x for p in pts])
#                 ay = np.array([p.actual_y for p in pts]); nx = np.array([p.nash_x for p in pts])
#                 ny = np.array([p.nash_y for p in pts]); lv = np.array([p.linear_vel for p in pts])
#                 av = np.array([p.angular_vel for p in pts]); pe = np.sqrt((ax - nx) ** 2 + (ay - ny) ** 2)

#                 ax_down, ay_down = self._downsample(ax, ay)
#                 traj_lines.append(PlotLine(ax_down, ay_down, f"R{rid} A", "-", traj_colors[i]))
#                 nx_down, ny_down = self._downsample(nx, ny)
#                 traj_lines.append(PlotLine(nx_down, ny_down, f"R{rid} N", "--", traj_colors[i], "o", 2, "none"))

#                 ts_rel = ts - start_time_offset
#                 ts_lv_down, lv_down = self._downsample(ts_rel, lv)
#                 vel_lines.append(PlotLine(ts_lv_down, lv_down, f"R{rid} Lin", "-", vel_colors[i * 2]))
#                 ts_av_down, av_down = self._downsample(ts_rel, av)
#                 vel_lines.append(PlotLine(ts_av_down, av_down, f"R{rid} Ang", "--", vel_colors[i * 2 + 1]))

#                 ts_pe_down, pe_down = self._downsample(ts_rel, pe)
#                 err_lines.append(PlotLine(ts_pe_down, pe_down, f"R{rid} Err", "-", err_colors[i]))

#             self.signals.plotting_data_ready.emit(
#                 ProcessedPlotData(True, trajectories=traj_lines, velocities=vel_lines, errors=err_lines, point_count=total_points)
#             )
#         except AttributeError as e:
#             if "lttb" in str(e):
#                 self.signals.plotting_data_ready.emit(ProcessedPlotData(False, f"`lttb`库函数调用错误: {e}. 请确保安装了正确的`lttb`包。"))
#             else:
#                 self.signals.plotting_data_ready.emit(ProcessedPlotData(False, f"处理数据时出错: {e}"))
#         except Exception as e:
#             import traceback
#             traceback.print_exc()
#             self.signals.plotting_data_ready.emit(ProcessedPlotData(False, f"处理数据时出错: {e}"))


# class GenericWorker(QRunnable):
#     def __init__(self, fn, signals: WorkerSignals, *args, **kwargs):
#         super().__init__()
#         self.fn = fn
#         self.signals = signals
#         self.args = args
#         self.kwargs = kwargs

#     def run(self):
#         try:
#             res = self.fn(*self.args, **self.kwargs)
#             fn_name = self.fn.__name__
#             if fn_name == "test_connection":
#                 self.signals.connection_result.emit(*res)
#             elif fn_name == "load_table_names":
#                 self.signals.table_names_loaded.emit(res)
#             elif fn_name == "get_time_range":
#                 self.signals.time_range_result.emit(*res)
#         except Exception as e:
#             print(f"工作线程错误: {e}")
#             if self.fn.__name__ == "test_connection":
#                 self.signals.connection_result.emit(False, str(e))


# class DatabaseManager:
#     def _get_connection_dict(self, db_params: DBParams) -> Dict[str, Any]:
#         h, u, p, d = db_params
#         hp_parts = h.replace("tcp://", "").split(":")
#         host_addr = hp_parts[0]
#         port = int(hp_parts[1]) if len(hp_parts) > 1 else 3306
#         return {
#             "host": host_addr, "port": port, "user": u, "password": p, "database": d, "connect_timeout": 10
#         }

#     def test_connection(self, h: str, u: str, p: str, d: str) -> Tuple[bool, str]:
#         conn = None
#         try:
#             conn = mysql.connector.connect(**self._get_connection_dict((h, u, p, d)))
#             return (True, "连接成功") if conn.is_connected() else (False, "连接失败")
#         except DB_Error as e:
#             return False, f"连接失败: {e.msg} (错误码: {e.errno})"
#         except Exception as e:
#             return False, f"连接失败: {e}"
#         finally:
#             if conn and conn.is_connected():
#                 conn.close()

#     def load_table_names(self, db_params: DBParams) -> List[str]:
#         conn = None
#         try:
#             conn = mysql.connector.connect(**self._get_connection_dict(db_params))
#             with conn.cursor() as c:
#                 c.execute("SHOW TABLES")
#                 return [""] + [t[0] for t in c.fetchall()]
#         except DB_Error as e:
#             print(f"加载表失败: {e}")
#             return [""]
#         finally:
#             if conn and conn.is_connected():
#                 conn.close()

#     def get_time_range(self, db_params: DBParams, table_name: str) -> Tuple[float, float, bool]:
#         if not table_name:
#             return 0.0, 0.0, False
#         conn = None
#         try:
#             conn = mysql.connector.connect(**self._get_connection_dict(db_params))
#             with conn.cursor() as c:
#                 c.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM `{table_name}` WHERE timestamp IS NOT NULL")
#                 result = c.fetchone()
#                 if result and result[0] is not None and result[1] is not None:
#                     return float(result[0]), float(result[1]), True
#                 return 0.0, 0.0, False
#         except DB_Error as e:
#             print(f"获取时间范围失败: {e}")
#             return 0.0, 0.0, False
#         finally:
#             if conn and conn.is_connected():
#                 conn.close()

#     def query_data(self, db_params: DBParams, table_name: str, start_ts: float, end_ts: float) -> Tuple[RobotDataMap, bool, str]:
#         if not table_name:
#             return {}, False, "未选择数据表"
#         conn = None
#         try:
#             conn = mysql.connector.connect(**self._get_connection_dict(db_params))
#             cache: RobotDataMap = {}
#             with conn.cursor(dictionary=True) as c:
#                 # [MODIFIED] 将 id 修改为 robot_id，以匹配目标格式。
#                 # 请确保你的数据库表结构中包含 robot_id 这一列。
#                 query = (
#                     f"SELECT timestamp, robot_id, actual_x, actual_y, nash_x, nash_y, linear_vel, angular_vel "
#                     f"FROM `{table_name}` WHERE timestamp BETWEEN %s AND %s ORDER BY robot_id, timestamp ASC"
#                 )
#                 c.execute(query, (start_ts, end_ts))
#                 for row in c.fetchall():
#                     # [MODIFIED] 使用 'robot_id' 作为机器人ID的键。
#                     rid = int(row["robot_id"])
#                     if rid not in cache:
#                         cache[rid] = []
#                     cache[rid].append(
#                         PlotDataPoint(
#                             timestamp=row.get("timestamp", 0.0),
#                             actual_x=row.get("actual_x", 0.0),
#                             actual_y=row.get("actual_y", 0.0),
#                             nash_x=row.get("nash_x", 0.0),
#                             nash_y=row.get("nash_y", 0.0),
#                             linear_vel=row.get("linear_vel", 0.0),
#                             angular_vel=row.get("angular_vel", 0.0),
#                         )
#                     )
#             return cache, True, "查询成功"
#         except DB_Error as e:
#             # 如果出现 "Unknown column 'robot_id'" 错误，请检查你的表结构。
#             print(f"查询失败: {e}")
#             return {}, False, f"查询失败: {e}"
#         finally:
#             if conn and conn.is_connected():
#                 conn.close()

# # =========================================================================
# # 3. MainWindow
# # =========================================================================
# class MainWindow(QMainWindow):
#     SLIDER_RESOLUTION = 10000
#     PLOT_DOWNSAMPLE_THRESHOLD = 4000
#     LIVE_UPDATE_INTERVAL = 2000

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.thread_pool = QThreadPool()
#         self.db_manager = DatabaseManager()
#         self.db_params: Optional[DBParams] = None
#         self.is_loading = False
#         self.min_table_timestamp = 0.0
#         self.max_table_timestamp = 0.0
#         self.is_connected = False
#         self.signals = WorkerSignals()
#         self.plot_interactors: Dict[str, DraggablePanAndZoom] = {}
#         self.use_chinese_labels = find_chinese_font()
#         self.setupUi()
#         self.connect_signals()
#         QTimer.singleShot(100, self.update_ui_state)

#     def setupUi(self):
#         self.setWindowTitle("高性能MySQL数据可视化")
#         self.resize(1800, 1000)
#         central_widget = QWidget(self)
#         self.setCentralWidget(central_widget)
#         main_layout = QVBoxLayout(central_widget)
#         self.setStatusBar(QStatusBar(self))
#         status_bar = self.statusBar()
#         self.progress_bar = QProgressBar(status_bar)
#         self.progress_bar.setMaximumWidth(200)
#         self.progress_bar.hide()
#         status_bar.addPermanentWidget(self.progress_bar)
#         top_panel = QWidget()
#         top_layout = QGridLayout(top_panel)
#         main_layout.addWidget(top_panel)

#         self.tab_widget = QTabWidget()

#         self.trajectory_fig, self.trajectory_ax = self._create_plot_tab("trajectory", "轨迹图", "世界坐标 X (m)", "世界坐标 Y (m)")
#         self.velocity_fig, self.velocity_ax = self._create_plot_tab("velocity", "速度图", "时间 (s)", "速度 (m/s, rad/s)")
#         self.error_fig, self.error_ax = self._create_plot_tab("error", "误差图", "时间 (s)", "位置误差 (m)")
        
#         self._setup_connection_widgets(top_layout)
#         self._setup_controls(top_layout)

#         main_layout.addWidget(self.tab_widget)


#     def _create_axis_range_widgets(self, plot_name: str) -> QWidget:
#         widget = QWidget()
#         layout = QHBoxLayout(widget)
#         layout.setContentsMargins(0, 5, 0, 5)

#         setattr(self, f"{plot_name}_xmin_edit", QLineEdit("-10"))
#         setattr(self, f"{plot_name}_xmax_edit", QLineEdit("10"))
#         setattr(self, f"{plot_name}_ymin_edit", QLineEdit("-10"))
#         setattr(self, f"{plot_name}_ymax_edit", QLineEdit("10"))

#         layout.addWidget(QLabel("X范围:"))
#         layout.addWidget(getattr(self, f"{plot_name}_xmin_edit"))
#         layout.addWidget(QLabel("到"))
#         layout.addWidget(getattr(self, f"{plot_name}_xmax_edit"))
#         layout.addSpacing(20)
#         layout.addWidget(QLabel("Y范围:"))
#         layout.addWidget(getattr(self, f"{plot_name}_ymin_edit"))
#         layout.addWidget(QLabel("到"))
#         layout.addWidget(getattr(self, f"{plot_name}_ymax_edit"))
#         layout.addStretch()

#         return widget

#     def _create_plot_tab(self, name: str, title_zh: str, xlabel_zh: str, ylabel_zh: str) -> Tuple[Figure, plt.Axes]:
#         container_widget = QWidget()
#         container_layout = QVBoxLayout(container_widget)
#         container_layout.setContentsMargins(0, 0, 0, 0)

#         range_widget = self._create_axis_range_widgets(name)
#         container_layout.addWidget(range_widget)

#         fig = Figure(dpi=100)
#         canvas = FigureCanvas(fig)
#         ax = fig.add_subplot(111)
#         ax.grid(True)
#         container_layout.addWidget(canvas)
#         container_layout.addWidget(NavigationToolbar2QT(canvas, container_widget))

#         title = title_zh if self.use_chinese_labels else name.capitalize()
#         xlabel = xlabel_zh if self.use_chinese_labels else f"{name.capitalize()} X"
#         ylabel = ylabel_zh if self.use_chinese_labels else f"{name.capitalize()} Y"

#         setattr(self, f"{name}_title_edit", QLineEdit(title))
#         setattr(self, f"{name}_xlabel_edit", QLineEdit(xlabel))
#         setattr(self, f"{name}_ylabel_edit", QLineEdit(ylabel))

#         self.plot_interactors[name] = DraggablePanAndZoom(fig, ax)

#         self.tab_widget.addTab(container_widget, title)
#         return fig, ax

#     def _setup_plot_settings_controls(self, layout: QGridLayout):
#         group_title = "绘图设置" if self.use_chinese_labels else "Plot Settings"
#         self.plot_settings_group = QGroupBox(group_title)
#         psl = QGridLayout(self.plot_settings_group)
        
#         bold_text = "粗体" if self.use_chinese_labels else "Bold"
#         title_size_text = "标题字号:" if self.use_chinese_labels else "Title Size:"
#         label_size_text = "标签字号:" if self.use_chinese_labels else "Label Size:"
#         line_width_text = "线宽:" if self.use_chinese_labels else "Line Width:"

#         self.title_size_spinbox = QSpinBox(); self.title_size_spinbox.setRange(8, 24); self.title_size_spinbox.setValue(12)
#         self.title_bold_checkbox = QCheckBox(bold_text); self.title_bold_checkbox.setChecked(True)
#         self.label_size_spinbox = QSpinBox(); self.label_size_spinbox.setRange(8, 20); self.label_size_spinbox.setValue(10)
#         self.label_bold_checkbox = QCheckBox(bold_text); self.label_bold_checkbox.setChecked(True)
#         self.line_width_spinbox = QSpinBox(); self.line_width_spinbox.setRange(1, 10); self.line_width_spinbox.setValue(2)

#         psl.addWidget(QLabel(title_size_text), 0, 0); psl.addWidget(self.title_size_spinbox, 0, 1); psl.addWidget(self.title_bold_checkbox, 0, 2)
#         psl.addWidget(QLabel(label_size_text), 0, 3); psl.addWidget(self.label_size_spinbox, 0, 4); psl.addWidget(self.label_bold_checkbox, 0, 5)
#         psl.addWidget(QLabel(line_width_text), 0, 6); psl.addWidget(self.line_width_spinbox, 0, 7)

#         def add_plot_specific_settings(name, row):
#             psl.addWidget(QLabel(f"{name.capitalize()}:"), row, 0)
#             psl.addWidget(getattr(self, f"{name}_title_edit"), row, 1, 1, 2)
#             psl.addWidget(QLabel("X轴:"), row, 3); psl.addWidget(getattr(self, f"{name}_xlabel_edit"), row, 4)
#             psl.addWidget(QLabel("Y轴:"), row, 5); psl.addWidget(getattr(self, f"{name}_ylabel_edit"), row, 6)

#         add_plot_specific_settings("trajectory", 1)
#         add_plot_specific_settings("velocity", 2)
#         add_plot_specific_settings("error", 3)
        
#         apply_text = "应用范围" if self.use_chinese_labels else "Apply Ranges"
#         auto_text = "自动范围" if self.use_chinese_labels else "Auto Range"
#         self.apply_ranges_button = QPushButton(apply_text)
#         self.auto_range_button = QPushButton(auto_text)
#         psl.addWidget(self.apply_ranges_button, 4, 1)
#         psl.addWidget(self.auto_range_button, 4, 2)

#         layout.addWidget(self.plot_settings_group, 4, 0, 1, 10)

#     def _setup_connection_widgets(self, layout: QGridLayout):
#         layout.addWidget(QLabel("主机:"), 0, 0); self.host_line_edit = QLineEdit("tcp://127.0.0.1:3306"); layout.addWidget(self.host_line_edit, 0, 1, 1, 2)
#         layout.addWidget(QLabel("用户:"), 0, 3); self.user_line_edit = QLineEdit("car_user"); layout.addWidget(self.user_line_edit, 0, 4)
#         layout.addWidget(QLabel("密码:"), 0, 5); self.pass_line_edit = QLineEdit("StrongPassword123!"); self.pass_line_edit.setEchoMode(QLineEdit.Password); layout.addWidget(self.pass_line_edit, 0, 6)
#         layout.addWidget(QLabel("数据库:"), 0, 7); self.db_line_edit = QLineEdit("car_tracking"); layout.addWidget(self.db_line_edit, 0, 8)
#         self.connect_button = QPushButton("连接"); layout.addWidget(self.connect_button, 0, 9)

#     def _setup_controls(self, layout: QGridLayout):
#         row = 1
#         layout.addWidget(QLabel("数据表:"), row, 0); self.table_combo_box = QComboBox(); layout.addWidget(self.table_combo_box, row, 1, 1, 2)
#         layout.addWidget(QLabel("开始:"), row, 3); self.start_time_edit = QDateTimeEdit(); self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss"); self.start_time_edit.setCalendarPopup(True); layout.addWidget(self.start_time_edit, row, 4, 1, 2)
#         layout.addWidget(QLabel("结束:"), row, 6); self.end_time_edit = QDateTimeEdit(QDateTime.currentDateTime()); self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss"); self.end_time_edit.setCalendarPopup(True); layout.addWidget(self.end_time_edit, row, 7, 1, 2)
#         self.now_button = QPushButton("现在"); layout.addWidget(self.now_button, row, 9)

#         row = 2
#         self.live_update_checkbox = QCheckBox("实时更新"); layout.addWidget(self.live_update_checkbox, row, 0)
#         self.start_range_slider = QSlider(Qt.Horizontal); self.start_range_slider.setRange(0, self.SLIDER_RESOLUTION); layout.addWidget(self.start_range_slider, row, 1, 1, 9)

#         row = 3
#         self.plot_button = QPushButton("绘图"); self.plot_button.setStyleSheet("QPushButton{background-color:#4CAF50;color:white;border-radius:5px;padding:6px}QPushButton:hover{background-color:#45a049}QPushButton:disabled{background-color:#cccccc}"); layout.addWidget(self.plot_button, row, 0)
#         self.end_range_slider = QSlider(Qt.Horizontal); self.end_range_slider.setRange(0, self.SLIDER_RESOLUTION); self.end_range_slider.setValue(self.SLIDER_RESOLUTION); layout.addWidget(self.end_range_slider, row, 1, 1, 9)
        
#         self._setup_plot_settings_controls(layout)

#     def connect_signals(self):
#         self.signals.connection_result.connect(self.handle_connection_result)
#         self.signals.table_names_loaded.connect(self.handle_table_names_loaded)
#         self.signals.time_range_result.connect(self.handle_time_range_result)
#         self.signals.plotting_data_ready.connect(self.handle_plotting_data_ready)

#         self.connect_button.clicked.connect(self.run_db_connect)
#         self.plot_button.clicked.connect(self.fetch_and_plot_data)
#         self.table_combo_box.currentTextChanged.connect(self.on_table_name_changed)
#         self.live_update_checkbox.stateChanged.connect(self.on_live_update_toggled)
#         self.now_button.clicked.connect(lambda: self.end_time_edit.setDateTime(QDateTime.currentDateTime()))
#         self.start_range_slider.valueChanged.connect(self.on_start_slider_changed)
#         self.end_range_slider.valueChanged.connect(self.on_end_slider_changed)

#         for w in [self.title_size_spinbox, self.label_size_spinbox, self.line_width_spinbox, self.title_bold_checkbox, self.label_bold_checkbox]:
#             if hasattr(w, "valueChanged"): w.valueChanged.connect(self.update_all_canvases)
#             else: w.stateChanged.connect(self.update_all_canvases)
        
#         for name in ["trajectory", "velocity", "error"]:
#             getattr(self, f"{name}_title_edit").textChanged.connect(self.update_all_canvases)
#             getattr(self, f"{name}_xlabel_edit").textChanged.connect(self.update_all_canvases)
#             getattr(self, f"{name}_ylabel_edit").textChanged.connect(self.update_all_canvases)

#         self.apply_ranges_button.clicked.connect(self.on_apply_ranges)
#         self.auto_range_button.clicked.connect(self.on_auto_range)

#         self.update_timer = QTimer(self)
#         self.update_timer.setInterval(self.LIVE_UPDATE_INTERVAL)
#         self.update_timer.timeout.connect(self.fetch_and_plot_data)

#     def on_apply_ranges(self):
#         current_tab_index = self.tab_widget.currentIndex()
#         if current_tab_index == -1: return
#         plot_name = ["trajectory", "velocity", "error"][current_tab_index]
#         ax = getattr(self, f"{plot_name}_ax")
        
#         try:
#             xmin = float(getattr(self, f"{plot_name}_xmin_edit").text())
#             xmax = float(getattr(self, f"{plot_name}_xmax_edit").text())
#             ymin = float(getattr(self, f"{plot_name}_ymin_edit").text())
#             ymax = float(getattr(self, f"{plot_name}_ymax_edit").text())
            
#             if xmin >= xmax or ymin >= ymax:
#                 QMessageBox.warning(self, "范围错误", "最小值必须小于最大值。")
#                 return

#             ax.set_xlim(xmin, xmax)
#             ax.set_ylim(ymin, ymax)
#             ax.figure.canvas.draw_idle()
#         except ValueError:
#             QMessageBox.warning(self, "输入错误", "请输入有效的数字作为坐标轴范围。")

#     def on_auto_range(self):
#         current_tab_index = self.tab_widget.currentIndex()
#         if current_tab_index == -1: return
#         plot_name = ["trajectory", "velocity", "error"][current_tab_index]
#         ax = getattr(self, f"{plot_name}_ax")
#         ax.autoscale(enable=True, axis="both")
#         ax.relim()
#         ax.figure.canvas.draw_idle()
#         self._update_plot_styles(ax.figure, ax, plot_name)

#     def _update_plot_styles(self, fig: Figure, ax: plt.Axes, name: str):
#         title_edit = getattr(self, f"{name}_title_edit")
#         xlabel_edit = getattr(self, f"{name}_xlabel_edit")
#         ylabel_edit = getattr(self, f"{name}_ylabel_edit")
        
#         title_font = {"size": self.title_size_spinbox.value(), "weight": "bold" if self.title_bold_checkbox.isChecked() else "normal"}
#         label_font = {"size": self.label_size_spinbox.value(), "weight": "bold" if self.label_bold_checkbox.isChecked() else "normal"}
        
#         ax.set_title(title_edit.text(), fontdict=title_font)
#         ax.set_xlabel(xlabel_edit.text(), fontdict=label_font)
#         ax.set_ylabel(ylabel_edit.text(), fontdict=label_font)
        
#         for line in ax.get_lines():
#             line.set_linewidth(self.line_width_spinbox.value())
            
#         if legend := ax.get_legend():
#             for text in legend.get_texts():
#                 text.set_fontsize(self.label_size_spinbox.value())
                
#         fig.tight_layout()
#         fig.canvas.draw_idle()
        
#         xmin, xmax = ax.get_xlim()
#         ymin, ymax = ax.get_ylim()
#         getattr(self, f"{name}_xmin_edit").setText(f"{xmin:.2f}")
#         getattr(self, f"{name}_xmax_edit").setText(f"{xmax:.2f}")
#         getattr(self, f"{name}_ymin_edit").setText(f"{ymin:.2f}")
#         getattr(self, f"{name}_ymax_edit").setText(f"{ymax:.2f}")

#     def update_all_canvases(self):
#         self._update_plot_styles(self.trajectory_fig, self.trajectory_ax, "trajectory")
#         self._update_plot_styles(self.velocity_fig, self.velocity_ax, "velocity")
#         self._update_plot_styles(self.error_fig, self.error_ax, "error")

#     def plot_all(self, data: ProcessedPlotData):
#         self._plot_from_package(self.trajectory_ax, data.trajectories)
#         self.trajectory_ax.set_aspect("equal", adjustable="box")
#         self._plot_from_package(self.velocity_ax, data.velocities)
#         self._plot_from_package(self.error_ax, data.errors)
#         self.update_all_canvases()

#     def _plot_from_package(self, ax: plt.Axes, lines: List[PlotLine]):
#         if not lines: return
#         for line in lines:
#             ax.plot(line.x_data, line.y_data, label=line.label, color=line.color, ls=line.style, marker=line.marker, ms=line.markersize, fillstyle=line.fillstyle)
#         ax.legend(prop={"size": self.label_size_spinbox.value()})
#         ax.relim()
#         ax.autoscale_view()

#     def update_ui_state(self):
#         has_time_range = self.min_table_timestamp > 0 and self.max_table_timestamp > 0
#         is_live_update_enabled = self.live_update_checkbox.isChecked()
#         can_interact = not self.is_loading
#         self.connect_button.setEnabled(not self.is_connected and can_interact)
#         for w in [self.host_line_edit, self.user_line_edit, self.pass_line_edit, self.db_line_edit]:
#             w.setEnabled(not self.is_connected and can_interact)
#         self.plot_settings_group.setEnabled(self.is_connected and can_interact)
#         self.table_combo_box.setEnabled(self.is_connected and can_interact)
#         can_plot = self.is_connected and has_time_range and can_interact
#         self.plot_button.setEnabled(can_plot and not is_live_update_enabled)
#         self.live_update_checkbox.setEnabled(can_plot)
#         self.start_time_edit.setEnabled(can_plot)
#         self.start_range_slider.setEnabled(can_plot)
#         self.end_time_edit.setEnabled(can_plot and not is_live_update_enabled)
#         self.end_range_slider.setEnabled(can_plot and not is_live_update_enabled)
#         self.now_button.setEnabled(can_plot and not is_live_update_enabled)

#     def set_loading_state(self, loading: bool, message: str = ""):
#         self.is_loading = loading
#         self.progress_bar.setVisible(loading)
#         self.progress_bar.setRange(0, 0 if loading else 1)
#         if message:
#             self.show_status_message(message)
#         self.update_ui_state()
#         QApplication.processEvents()

#     def run_db_connect(self):
#         self.set_loading_state(True, "连接中...")
#         h = self.host_line_edit.text()
#         u = self.user_line_edit.text()
#         p = self.pass_line_edit.text()
#         d = self.db_line_edit.text()
#         if not all([h, u, d]):
#             QMessageBox.warning(self, "输入错误", "请填写主机、用户和数据库字段。")
#             self.set_loading_state(False)
#             return
#         worker = GenericWorker(self.db_manager.test_connection, self.signals, h, u, p, d)
#         self.thread_pool.start(worker)

#     def fetch_and_plot_data(self):
#         if self.is_loading: return
#         if not self.db_params:
#             if not self.live_update_checkbox.isChecked(): QMessageBox.warning(self, "错误", "未连接到数据库。")
#             return
#         if self.live_update_checkbox.isChecked(): self.end_time_edit.setDateTime(QDateTime.currentDateTime())
#         table_name = self.table_combo_box.currentText()
#         if not table_name:
#             if not self.live_update_checkbox.isChecked(): QMessageBox.warning(self, "提示", "请选择一张数据表。")
#             return
#         start_ts = self.start_time_edit.dateTime().toSecsSinceEpoch()
#         end_ts = self.end_time_edit.dateTime().toSecsSinceEpoch()
#         if start_ts > end_ts:
#             if not self.live_update_checkbox.isChecked(): QMessageBox.warning(self, "错误", "开始时间不能晚于结束时间。")
#             return
#         self.set_loading_state(True, "查询和处理数据中...")
#         worker = QueryAndProcessWorker(self.db_manager, self.db_params, self.signals, table_name, start_ts, end_ts)
#         self.thread_pool.start(worker)
        
#     def handle_connection_result(self, success: bool, message: str):
#         if success:
#             self.is_connected = True
#             self.db_params = (self.host_line_edit.text(), self.user_line_edit.text(), self.pass_line_edit.text(), self.db_line_edit.text())
#             QMessageBox.information(self, "成功", message)
#             self.set_loading_state(True, "加载数据表...")
#             worker = GenericWorker(self.db_manager.load_table_names, self.signals, self.db_params)
#             self.thread_pool.start(worker)
#         else:
#             self.is_connected = False
#             self.db_params = None
#             QMessageBox.critical(self, "错误", message)
#             self.set_loading_state(False, "连接失败")

#     def handle_table_names_loaded(self, tables: List[str]):
#         self.table_combo_box.clear()
#         self.table_combo_box.addItems(tables)
#         if len(tables) > 1: self.table_combo_box.setCurrentIndex(1)
#         self.set_loading_state(False, "表加载完成")

#     def handle_time_range_result(self, min_ts: float, max_ts: float, success: bool):
#         self.min_table_timestamp = 0.0
#         self.max_table_timestamp = 0.0
#         if success:
#             self.min_table_timestamp, self.max_table_timestamp = min_ts, max_ts
#             self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(min_ts)))
#             self.end_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(max_ts)))
#             self.start_range_slider.setValue(0)
#             self.end_range_slider.setValue(self.SLIDER_RESOLUTION)
#             self.set_loading_state(False, "时间范围已获取")
#             self.fetch_and_plot_data()
#         else:
#             self.set_loading_state(False, "无法获取时间范围")
#             self.clear_all_plots()

#     def handle_plotting_data_ready(self, data: ProcessedPlotData):
#         self.clear_all_plots()
#         if data.success:
#             self.plot_all(data)
#             msg = f"绘图完成。处理点数: {data.point_count}" + (f" | {data.message}" if data.message else "")
#             self.show_status_message(msg, 5000)
#         else:
#             msg = data.message or "绘图失败"
#             self.show_status_message(msg, 5000)
#             QMessageBox.warning(self, "绘图信息", msg)
#         self.set_loading_state(False)
    
#     def on_table_name_changed(self, table_name: str):
#         self.clear_all_plots()
#         self.min_table_timestamp, self.max_table_timestamp = 0.0, 0.0
#         if table_name and self.db_params:
#             self.set_loading_state(True, f"获取 '{table_name}' 时间范围...")
#             worker = GenericWorker(self.db_manager.get_time_range, self.signals, self.db_params, table_name)
#             self.thread_pool.start(worker)
#         else:
#             self.update_ui_state()

#     def on_live_update_toggled(self, state: int):
#         self.update_ui_state()
#         if state == Qt.Checked:
#             self.update_timer.start()
#             self.end_time_edit.setDateTime(QDateTime.currentDateTime())
#             self.fetch_and_plot_data()
#         else:
#             self.update_timer.stop()

#     def on_start_slider_changed(self, value: int):
#         if self.max_table_timestamp <= self.min_table_timestamp: return
#         ts = self.min_table_timestamp + (self.max_table_timestamp - self.min_table_timestamp) * (value / self.SLIDER_RESOLUTION)
#         self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(ts)))
#         if value > self.end_range_slider.value(): self.end_range_slider.setValue(value)

#     def on_end_slider_changed(self, value: int):
#         if self.max_table_timestamp <= self.min_table_timestamp: return
#         ts = self.min_table_timestamp + (self.max_table_timestamp - self.min_table_timestamp) * (value / self.SLIDER_RESOLUTION)
#         self.end_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(ts)))
#         if value < self.start_range_slider.value(): self.start_range_slider.setValue(value)

#     def clear_all_plots(self):
#         for ax in [self.trajectory_ax, self.velocity_ax, self.error_ax]:
#             ax.clear()
#             ax.grid(True)
#             if ax.get_legend(): ax.get_legend().remove()
#         for fig in [self.trajectory_fig, self.velocity_fig, self.error_fig]:
#             fig.canvas.draw_idle()

#     def show_status_message(self, message: str, timeout: int = 0):
#         self.statusBar().showMessage(message, timeout)

#     def closeEvent(self, event):
#         self.thread_pool.waitForDone(-1)
#         event.accept()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     app.setStyle("Fusion")
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec_())