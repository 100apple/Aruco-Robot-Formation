import sys
import math
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import mysql.connector
from mysql.connector import Error as DB_Error

import plotly.graph_objects as go
from plotly_resampler import FigureResampler
import plotly.colors as pcolors

from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QMessageBox, QTabWidget, QSpinBox,
    QDateTimeEdit, QSlider, QCheckBox, QGroupBox, QProgressBar, QStatusBar,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRunnable, QThreadPool, QDateTime, QTimer

# =========================================================================
# 0. Plotly Widget Helper (MODIFIED)
# =========================================================================
class PlotlyWidget(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHtml(self._get_html_template(), QUrl("about:blank"))
        # The figure object is now managed by MainWindow, not created here.
        self.figure = None

    def _get_html_template(self) -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
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
        self.figure = fig
        if fig:
            fig_json = fig.to_json()
            self.page().runJavaScript(f"updatePlot({fig_json})")
        else:
            self.clear()

    def clear(self):
        # FIX: A simpler clear method that doesn't instantiate a figure.
        self.page().runJavaScript("Plotly.react('plotdiv', [], {}, {responsive: true});")

# =========================================================================
# 1. Data Structures & Plotting Data Packages (Unchanged)
# =========================================================================
@dataclass
class PlotDataPoint:
    timestamp: float; actual_x: float; actual_y: float
    nash_x: float; nash_y: float; linear_vel: float; angular_vel: float

@dataclass
class PlotLine:
    x_data: np.ndarray; y_data: np.ndarray
    label: str; dash: str; color: Any
    mode: str = "lines"; marker_symbol: str = "circle"; marker_size: int = 4

@dataclass
class ProcessedPlotData:
    success: bool; message: str = ""; point_count: int = 0
    trajectories: List[PlotLine] = field(default_factory=list)
    velocities: List[PlotLine] = field(default_factory=list)
    errors: List[PlotLine] = field(default_factory=list)

RobotDataMap = Dict[int, List[PlotDataPoint]]
DBParams = Tuple[str, str, str, str]

# =========================================================================
# 2. Worker/Manager Architecture (Unchanged)
# =========================================================================
class WorkerSignals(QObject):
    connection_result = pyqtSignal(bool, str)
    table_names_loaded = pyqtSignal(list)
    time_range_result = pyqtSignal(float, float, bool)
    plotting_data_ready = pyqtSignal(ProcessedPlotData)

class QueryAndProcessWorker(QRunnable):
    def __init__(self, db_manager, db_params: DBParams, signals: WorkerSignals, table_name: str, start_ts: float, end_ts: float):
        super().__init__()
        self.db_manager = db_manager; self.db_params = db_params
        self.signals = signals; self.table_name = table_name
        self.start_ts = start_ts; self.end_ts = end_ts

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
            
            color_sequence = pcolors.qualitative.Plotly
            all_timestamps = [p.timestamp for pts in raw_data.values() for p in pts]
            start_time_offset = min(all_timestamps) if all_timestamps else 0

            for i, (rid, pts) in enumerate(raw_data.items()):
                if not pts: continue
                
                ts = np.array([p.timestamp for p in pts]); ax = np.array([p.actual_x for p in pts])
                ay = np.array([p.actual_y for p in pts]); nx = np.array([p.nash_x for p in pts])
                ny = np.array([p.nash_y for p in pts]); lv = np.array([p.linear_vel for p in pts])
                av = np.array([p.angular_vel for p in pts]); pe = np.sqrt((ax - nx) ** 2 + (ay - ny) ** 2)

                color = color_sequence[i % len(color_sequence)]
                vel_color_1 = color_sequence[(i * 2) % len(color_sequence)]
                vel_color_2 = color_sequence[(i * 2 + 1) % len(color_sequence)]
                
                traj_lines.append(PlotLine(ax, ay, f"R{rid} Actual", "solid", color))
                traj_lines.append(PlotLine(nx, ny, f"R{rid} Nash", "dash", color, mode="lines+markers", marker_symbol="circle-open", marker_size=6))

                ts_rel = ts - start_time_offset
                vel_lines.append(PlotLine(ts_rel, lv, f"R{rid} Linear Vel", "solid", vel_color_1))
                vel_lines.append(PlotLine(ts_rel, av, f"R{rid} Angular Vel", "dash", vel_color_2))
                err_lines.append(PlotLine(ts_rel, pe, f"R{rid} Position Err", "solid", color))

            self.signals.plotting_data_ready.emit(
                ProcessedPlotData(True, trajectories=traj_lines, velocities=vel_lines, errors=err_lines, point_count=total_points)
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            self.signals.plotting_data_ready.emit(ProcessedPlotData(False, f"处理数据时出错: {e}"))
# ... (GenericWorker and DatabaseManager unchanged)
class GenericWorker(QRunnable):
    def __init__(self, fn, signals: WorkerSignals, *args, **kwargs): super().__init__(); self.fn = fn; self.signals = signals; self.args = args; self.kwargs = kwargs
    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            fn_name = self.fn.__name__
            if fn_name == "test_connection": self.signals.connection_result.emit(*res)
            elif fn_name == "load_table_names": self.signals.table_names_loaded.emit(res)
            elif fn_name == "get_time_range": self.signals.time_range_result.emit(*res)
        except Exception as e:
            print(f"工作线程错误: {e}")
            if self.fn.__name__ == "test_connection": self.signals.connection_result.emit(False, str(e))
class DatabaseManager:
    def _get_connection_dict(self, db_params: DBParams) -> Dict[str, Any]:
        h, u, p, d = db_params; hp_parts = h.replace("tcp://", "").split(":"); host_addr = hp_parts[0]; port = int(hp_parts[1]) if len(hp_parts) > 1 else 3306
        return {"host": host_addr, "port": port, "user": u, "password": p, "database": d, "connect_timeout": 10}
    def test_connection(self, h: str, u: str, p: str, d: str) -> Tuple[bool, str]:
        conn = None
        try: conn = mysql.connector.connect(**self._get_connection_dict((h, u, p, d))); return (True, "连接成功") if conn.is_connected() else (False, "连接失败")
        except DB_Error as e: return False, f"连接失败: {e.msg} (错误码: {e.errno})"
        except Exception as e: return False, f"连接失败: {e}"
        finally:
            if conn and conn.is_connected(): conn.close()
    def load_table_names(self, db_params: DBParams) -> List[str]:
        conn = None
        try:
            conn = mysql.connector.connect(**self._get_connection_dict(db_params))
            with conn.cursor() as c: c.execute("SHOW TABLES"); return [""] + [t[0] for t in c.fetchall()]
        except DB_Error as e: print(f"加载表失败: {e}"); return [""]
        finally:
            if conn and conn.is_connected(): conn.close()
    def get_time_range(self, db_params: DBParams, table_name: str) -> Tuple[float, float, bool]:
        if not table_name: return 0.0, 0.0, False
        conn = None
        try:
            conn = mysql.connector.connect(**self._get_connection_dict(db_params))
            with conn.cursor() as c:
                c.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM `{table_name}` WHERE timestamp IS NOT NULL")
                result = c.fetchone()
                if result and result[0] is not None and result[1] is not None: return float(result[0]), float(result[1]), True
                return 0.0, 0.0, False
        except DB_Error as e: print(f"获取时间范围失败: {e}"); return 0.0, 0.0, False
        finally:
            if conn and conn.is_connected(): conn.close()
    def query_data(self, db_params: DBParams, table_name: str, start_ts: float, end_ts: float) -> Tuple[RobotDataMap, bool, str]:
        if not table_name: return {}, False, "未选择数据表"
        conn = None
        try:
            conn = mysql.connector.connect(**self._get_connection_dict(db_params))
            cache: RobotDataMap = {}
            with conn.cursor(dictionary=True) as c:
                query = (f"SELECT timestamp, robot_id, actual_x, actual_y, nash_x, nash_y, linear_vel, angular_vel FROM `{table_name}` WHERE timestamp BETWEEN %s AND %s ORDER BY robot_id, timestamp ASC")
                c.execute(query, (start_ts, end_ts))
                for row in c.fetchall():
                    rid = int(row["robot_id"])
                    if rid not in cache: cache[rid] = []
                    cache[rid].append(PlotDataPoint(timestamp=float(row.get("timestamp", 0.0)), actual_x=float(row.get("actual_x", 0.0)), actual_y=float(row.get("actual_y", 0.0)), nash_x=float(row.get("nash_x", 0.0)), nash_y=float(row.get("nash_y", 0.0)), linear_vel=float(row.get("linear_vel", 0.0)), angular_vel=float(row.get("angular_vel", 0.0)),))
            return cache, True, "查询成功"
        except DB_Error as e: print(f"查询失败: {e}"); return {}, False, f"查询失败: {e}"
        finally:
            if conn and conn.is_connected(): conn.close()

# =========================================================================
# 3. MainWindow (MODIFIED)
# =========================================================================
class MainWindow(QMainWindow):
    SLIDER_RESOLUTION = 10000; PLOT_DOWNSAMPLE_THRESHOLD = 2000; LIVE_UPDATE_INTERVAL = 2000
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread_pool = QThreadPool(); self.db_manager = DatabaseManager(); self.db_params: Optional[DBParams] = None
        self.is_loading = False; self.min_table_timestamp = 0.0; self.max_table_timestamp = 0.0
        self.is_connected = False; self.signals = WorkerSignals(); self.plots = {}
        self.setupUi(); self.connect_signals(); QTimer.singleShot(100, self.update_ui_state)

    def _create_plot_tab(self, name: str, title: str, xlabel: str, ylabel: str):
        container_widget = QWidget(); layout = QVBoxLayout(container_widget); layout.setContentsMargins(5, 5, 5, 5)
        range_widget = QWidget(); range_layout = QHBoxLayout(range_widget); range_layout.setContentsMargins(0, 0, 0, 0)
        xmin_edit = QLineEdit("-10"); xmax_edit = QLineEdit("10"); ymin_edit = QLineEdit("-10"); ymax_edit = QLineEdit("10")
        range_layout.addWidget(QLabel("X范围:")); range_layout.addWidget(xmin_edit); range_layout.addWidget(QLabel("到")); range_layout.addWidget(xmax_edit)
        range_layout.addSpacing(20); range_layout.addWidget(QLabel("Y范围:")); range_layout.addWidget(ymin_edit); range_layout.addWidget(QLabel("到")); range_layout.addWidget(ymax_edit)
        range_layout.addStretch(); layout.addWidget(range_widget)
        plot_widget = PlotlyWidget(); layout.addWidget(plot_widget)
        
        # FIX: Instantiate the correct figure type from the beginning.
        if name == "trajectory":
            fig = go.Figure()
        else:
            fig = FigureResampler()
            
        self.plots[name] = {"fig": fig, "widget": plot_widget, "title_edit": QLineEdit(title), "xlabel_edit": QLineEdit(xlabel), "ylabel_edit": QLineEdit(ylabel), "xmin_edit": xmin_edit, "xmax_edit": xmax_edit, "ymin_edit": ymin_edit, "ymax_edit": ymax_edit}
        self.tab_widget.addTab(container_widget, title)

    # FINAL FIX is here
    def _update_figure_from_data(self, name: str, lines: List[PlotLine]):
        plot_info = self.plots[name]
        
        # FIX 1: Create a new figure of the correct type each time we update.
        if name == "trajectory":
            fig = go.Figure()
        else:
            fig = FigureResampler()
            
        line_width = self.line_width_spinbox.value()
        
        for line in lines:
            trace = go.Scattergl(
                x=line.x_data, y=line.y_data, name=line.label, mode=line.mode,
                line=dict(color=line.color, width=line_width, dash=line.dash),
                marker=dict(symbol=line.marker_symbol, size=line.marker_size)
            )
            
            # FIX 2: Check the FIGURE's type to decide how to add the trace.
            if isinstance(fig, FigureResampler):
                # Only use resampling for FigureResampler instances (velocity, error plots)
                fig.add_trace(trace, max_n_samples=self.PLOT_DOWNSAMPLE_THRESHOLD)
            else:
                # Use the standard method for regular go.Figure instances (trajectory plot)
                fig.add_trace(trace)
        
        if name == "trajectory":
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        plot_info["fig"] = fig

    # FINAL FIX is also here
    def clear_all_plots(self):
        for name, plot_info in self.plots.items():
            plot_info["widget"].clear()
            # FIX: Re-initialize with the correct empty figure type.
            if name == "trajectory":
                plot_info["fig"] = go.Figure()
            else:
                plot_info["fig"] = FigureResampler()

    # --- Rest of the code is unchanged from your last correct version ---
    def setupUi(self):
        self.setWindowTitle("数据可视化 (Plotly-Resampler版)"); self.resize(1800, 1000); central_widget = QWidget(self); self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget); self.setStatusBar(QStatusBar(self)); status_bar = self.statusBar()
        self.progress_bar = QProgressBar(status_bar); self.progress_bar.setMaximumWidth(200); self.progress_bar.hide(); status_bar.addPermanentWidget(self.progress_bar)
        top_panel = QWidget(); top_layout = QGridLayout(top_panel); main_layout.addWidget(top_panel); self.tab_widget = QTabWidget()
        self._create_plot_tab("trajectory", "轨迹图", "世界坐标 X (m)", "世界坐标 Y (m)"); self._create_plot_tab("velocity", "速度图", "时间 (s)", "速度 (m/s, rad/s)"); self._create_plot_tab("error", "误差图", "时间 (s)", "位置误差 (m)")
        self._setup_connection_widgets(top_layout); self._setup_controls(top_layout); main_layout.addWidget(self.tab_widget)
    def _setup_plot_settings_controls(self, layout: QGridLayout):
        self.plot_settings_group = QGroupBox("绘图设置"); psl = QGridLayout(self.plot_settings_group)
        self.title_size_spinbox = QSpinBox(); self.title_size_spinbox.setRange(8, 30); self.title_size_spinbox.setValue(16)
        self.label_size_spinbox = QSpinBox(); self.label_size_spinbox.setRange(8, 24); self.label_size_spinbox.setValue(12)
        self.line_width_spinbox = QSpinBox(); self.line_width_spinbox.setRange(1, 10); self.line_width_spinbox.setValue(2)
        self.legend_size_spinbox = QSpinBox(); self.legend_size_spinbox.setRange(8, 20); self.legend_size_spinbox.setValue(11)
        psl.addWidget(QLabel("标题字号:"), 0, 0); psl.addWidget(self.title_size_spinbox, 0, 1); psl.addWidget(QLabel("标签字号:"), 0, 2); psl.addWidget(self.label_size_spinbox, 0, 3)
        psl.addWidget(QLabel("图例字号:"), 0, 4); psl.addWidget(self.legend_size_spinbox, 0, 5); psl.addWidget(QLabel("线宽:"), 0, 6); psl.addWidget(self.line_width_spinbox, 0, 7)
        def add_plot_specific_settings(name, row):
            plot_widgets = self.plots[name]; psl.addWidget(QLabel(f"{name.capitalize()} 标题:"), row, 0); psl.addWidget(plot_widgets["title_edit"], row, 1, 1, 2)
            psl.addWidget(QLabel("X轴:"), row, 3); psl.addWidget(plot_widgets["xlabel_edit"], row, 4); psl.addWidget(QLabel("Y轴:"), row, 5); psl.addWidget(plot_widgets["ylabel_edit"], row, 6)
        add_plot_specific_settings("trajectory", 1); add_plot_specific_settings("velocity", 2); add_plot_specific_settings("error", 3)
        self.apply_ranges_button = QPushButton("应用范围"); self.auto_range_button = QPushButton("自动范围"); self.redraw_button = QPushButton("重绘样式")
        psl.addWidget(self.apply_ranges_button, 4, 1); psl.addWidget(self.auto_range_button, 4, 2); psl.addWidget(self.redraw_button, 4, 3); layout.addWidget(self.plot_settings_group, 4, 0, 1, 10)
    def _setup_connection_widgets(self, layout: QGridLayout):
        layout.addWidget(QLabel("主机:"), 0, 0); self.host_line_edit = QLineEdit("tcp://127.0.0.1:3306"); layout.addWidget(self.host_line_edit, 0, 1, 1, 2)
        layout.addWidget(QLabel("用户:"), 0, 3); self.user_line_edit = QLineEdit("car_user"); layout.addWidget(self.user_line_edit, 0, 4)
        layout.addWidget(QLabel("密码:"), 0, 5); self.pass_line_edit = QLineEdit("StrongPassword123!"); self.pass_line_edit.setEchoMode(QLineEdit.Password); layout.addWidget(self.pass_line_edit, 0, 6)
        layout.addWidget(QLabel("数据库:"), 0, 7); self.db_line_edit = QLineEdit("car_tracking"); layout.addWidget(self.db_line_edit, 0, 8); self.connect_button = QPushButton("连接"); layout.addWidget(self.connect_button, 0, 9)
    def _setup_controls(self, layout: QGridLayout):
        row = 1; layout.addWidget(QLabel("数据表:"), row, 0); self.table_combo_box = QComboBox(); layout.addWidget(self.table_combo_box, row, 1, 1, 2)
        layout.addWidget(QLabel("开始:"), row, 3); self.start_time_edit = QDateTimeEdit(); self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss"); self.start_time_edit.setCalendarPopup(True); layout.addWidget(self.start_time_edit, row, 4, 1, 2)
        layout.addWidget(QLabel("结束:"), row, 6); self.end_time_edit = QDateTimeEdit(QDateTime.currentDateTime()); self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss"); self.end_time_edit.setCalendarPopup(True); layout.addWidget(self.end_time_edit, row, 7, 1, 2)
        self.now_button = QPushButton("现在"); layout.addWidget(self.now_button, row, 9); row = 2; self.live_update_checkbox = QCheckBox("实时更新"); layout.addWidget(self.live_update_checkbox, row, 0)
        self.start_range_slider = QSlider(Qt.Horizontal); self.start_range_slider.setRange(0, self.SLIDER_RESOLUTION); layout.addWidget(self.start_range_slider, row, 1, 1, 9); row = 3
        self.plot_button = QPushButton("绘图"); self.plot_button.setStyleSheet("QPushButton{background-color:#4CAF50;color:white;border-radius:5px;padding:6px}QPushButton:hover{background-color:#45a049}QPushButton:disabled{background-color:#cccccc}"); layout.addWidget(self.plot_button, row, 0)
        self.end_range_slider = QSlider(Qt.Horizontal); self.end_range_slider.setRange(0, self.SLIDER_RESOLUTION); self.end_range_slider.setValue(self.SLIDER_RESOLUTION); layout.addWidget(self.end_range_slider, row, 1, 1, 9); self._setup_plot_settings_controls(layout)
    def connect_signals(self):
        self.signals.connection_result.connect(self.handle_connection_result); self.signals.table_names_loaded.connect(self.handle_table_names_loaded); self.signals.time_range_result.connect(self.handle_time_range_result); self.signals.plotting_data_ready.connect(self.handle_plotting_data_ready)
        self.connect_button.clicked.connect(self.run_db_connect); self.plot_button.clicked.connect(self.fetch_and_plot_data); self.table_combo_box.currentTextChanged.connect(self.on_table_name_changed); self.live_update_checkbox.stateChanged.connect(self.on_live_update_toggled)
        self.now_button.clicked.connect(lambda: self.end_time_edit.setDateTime(QDateTime.currentDateTime())); self.start_range_slider.valueChanged.connect(self.on_start_slider_changed); self.end_range_slider.valueChanged.connect(self.on_end_slider_changed)
        self.redraw_button.clicked.connect(self.update_all_plot_styles); self.apply_ranges_button.clicked.connect(self.on_apply_ranges); self.auto_range_button.clicked.connect(self.on_auto_range)
        self.update_timer = QTimer(self); self.update_timer.setInterval(self.LIVE_UPDATE_INTERVAL); self.update_timer.timeout.connect(self.fetch_and_plot_data)
    def plot_data(self, data: ProcessedPlotData):
        self._update_figure_from_data("trajectory", data.trajectories); self._update_figure_from_data("velocity", data.velocities); self._update_figure_from_data("error", data.errors); self.update_all_plot_styles()
    def update_all_plot_styles(self):
        for name, plot_info in self.plots.items():
            fig = plot_info["fig"]; widget = plot_info["widget"]
            fig.update_layout(title_text=plot_info["title_edit"].text(), xaxis_title=plot_info["xlabel_edit"].text(), yaxis_title=plot_info["ylabel_edit"].text(), title_font_size=self.title_size_spinbox.value(), xaxis_title_font_size=self.label_size_spinbox.value(), yaxis_title_font_size=self.label_size_spinbox.value(), legend_font_size=self.legend_size_spinbox.value(), legend=dict(x=1, y=1, xanchor='right', yanchor='top', bgcolor='rgba(255,255,255,0.5)'), margin=dict(l=50, r=20, t=50, b=40))
            widget.update_figure(fig)
    def on_apply_ranges(self):
        current_tab_index = self.tab_widget.currentIndex();
        if current_tab_index == -1: return
        plot_name = list(self.plots.keys())[current_tab_index]; plot_info = self.plots[plot_name]
        try:
            xmin = float(plot_info["xmin_edit"].text()); xmax = float(plot_info["xmax_edit"].text()); ymin = float(plot_info["ymin_edit"].text()); ymax = float(plot_info["ymax_edit"].text())
            if xmin >= xmax or ymin >= ymax: QMessageBox.warning(self, "范围错误", "最小值必须小于最大值."); return
            fig = plot_info["fig"]; fig.update_xaxes(range=[xmin, xmax]); fig.update_yaxes(range=[ymin, ymax]); plot_info["widget"].update_figure(fig)
        except ValueError: QMessageBox.warning(self, "输入错误", "请输入有效的数字作为坐标轴范围.")
    def on_auto_range(self):
        current_tab_index = self.tab_widget.currentIndex();
        if current_tab_index == -1: return
        plot_name = list(self.plots.keys())[current_tab_index]; plot_info = self.plots[plot_name]; fig = plot_info["fig"]
        fig.update_xaxes(autorange=True); fig.update_yaxes(autorange=True); plot_info["widget"].update_figure(fig)
    def handle_plotting_data_ready(self, data: ProcessedPlotData):
        self.clear_all_plots()
        if data.success:
            self.plot_data(data); msg = f"绘图完成。处理点数: {data.point_count}" + (f" | {data.message}" if data.message else ""); self.show_status_message(msg, 5000)
        else: msg = data.message or "绘图失败"; self.show_status_message(msg, 5000); QMessageBox.warning(self, "绘图信息", msg)
        self.set_loading_state(False)
    def on_table_name_changed(self, table_name: str):
        self.clear_all_plots(); self.min_table_timestamp, self.max_table_timestamp = 0.0, 0.0
        if table_name and self.db_params: self.set_loading_state(True, f"获取 '{table_name}' 时间范围..."); worker = GenericWorker(self.db_manager.get_time_range, self.signals, self.db_params, table_name); self.thread_pool.start(worker)
        else: self.update_ui_state()
    def update_ui_state(self):
        has_time_range = self.min_table_timestamp > 0 and self.max_table_timestamp > 0; is_live_update_enabled = self.live_update_checkbox.isChecked(); can_interact = not self.is_loading
        self.connect_button.setEnabled(not self.is_connected and can_interact);
        for w in [self.host_line_edit, self.user_line_edit, self.pass_line_edit, self.db_line_edit]: w.setEnabled(not self.is_connected and can_interact)
        self.plot_settings_group.setEnabled(self.is_connected and can_interact); self.table_combo_box.setEnabled(self.is_connected and can_interact); can_plot = self.is_connected and has_time_range and can_interact
        self.plot_button.setEnabled(can_plot and not is_live_update_enabled); self.live_update_checkbox.setEnabled(can_plot); self.start_time_edit.setEnabled(can_plot); self.start_range_slider.setEnabled(can_plot)
        self.end_time_edit.setEnabled(can_plot and not is_live_update_enabled); self.end_range_slider.setEnabled(can_plot and not is_live_update_enabled); self.now_button.setEnabled(can_plot and not is_live_update_enabled)
    def set_loading_state(self, loading: bool, message: str = ""):
        self.is_loading = loading; self.progress_bar.setVisible(loading); self.progress_bar.setRange(0, 0 if loading else 1);
        if message: self.show_status_message(message)
        self.update_ui_state(); QApplication.processEvents()
    def run_db_connect(self):
        self.set_loading_state(True, "连接中..."); h = self.host_line_edit.text(); u = self.user_line_edit.text(); p = self.pass_line_edit.text(); d = self.db_line_edit.text()
        if not all([h, u, d]): QMessageBox.warning(self, "输入错误", "请填写主机、用户和数据库字段。"); self.set_loading_state(False); return
        worker = GenericWorker(self.db_manager.test_connection, self.signals, h, u, p, d); self.thread_pool.start(worker)
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
        start_ts = self.start_time_edit.dateTime().toSecsSinceEpoch(); end_ts = self.end_time_edit.dateTime().toSecsSinceEpoch()
        if start_ts > end_ts:
            if not self.live_update_checkbox.isChecked(): QMessageBox.warning(self, "错误", "开始时间不能晚于结束时间。")
            return
        self.set_loading_state(True, "查询和处理数据中..."); worker = QueryAndProcessWorker(self.db_manager, self.db_params, self.signals, table_name, start_ts, end_ts); self.thread_pool.start(worker)
    def handle_connection_result(self, success: bool, message: str):
        if success:
            self.is_connected = True; self.db_params = (self.host_line_edit.text(), self.user_line_edit.text(), self.pass_line_edit.text(), self.db_line_edit.text())
            QMessageBox.information(self, "成功", message); self.set_loading_state(True, "加载数据表..."); worker = GenericWorker(self.db_manager.load_table_names, self.signals, self.db_params); self.thread_pool.start(worker)
        else: self.is_connected = False; self.db_params = None; QMessageBox.critical(self, "错误", message); self.set_loading_state(False, "连接失败")
    def handle_table_names_loaded(self, tables: List[str]):
        self.table_combo_box.clear(); self.table_combo_box.addItems(tables);
        if len(tables) > 1: self.table_combo_box.setCurrentIndex(1)
        self.set_loading_state(False, "表加载完成")
    def handle_time_range_result(self, min_ts: float, max_ts: float, success: bool):
        self.min_table_timestamp = 0.0; self.max_table_timestamp = 0.0
        if success:
            self.min_table_timestamp, self.max_table_timestamp = min_ts, max_ts; self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(min_ts))); self.end_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(max_ts)))
            self.start_range_slider.setValue(0); self.end_range_slider.setValue(self.SLIDER_RESOLUTION); self.set_loading_state(False, "时间范围已获取")
            if self.table_combo_box.currentText(): self.fetch_and_plot_data()
        else: self.set_loading_state(False, "无法获取时间范围"); self.clear_all_plots()
    def on_live_update_toggled(self, state: int):
        self.update_ui_state()
        if state == Qt.Checked: self.update_timer.start(); self.end_time_edit.setDateTime(QDateTime.currentDateTime()); self.fetch_and_plot_data()
        else: self.update_timer.stop()
    def on_start_slider_changed(self, value: int):
        if self.max_table_timestamp <= self.min_table_timestamp: return
        ts = self.min_table_timestamp + (self.max_table_timestamp - self.min_table_timestamp) * (value / self.SLIDER_RESOLUTION); self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(ts)))
        if value > self.end_range_slider.value(): self.end_range_slider.setValue(value)
    def on_end_slider_changed(self, value: int):
        if self.max_table_timestamp <= self.min_table_timestamp: return
        ts = self.min_table_timestamp + (self.max_table_timestamp - self.min_table_timestamp) * (value / self.SLIDER_RESOLUTION); self.end_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(ts)))
        if value < self.start_range_slider.value(): self.start_range_slider.setValue(value)
    def show_status_message(self, message: str, timeout: int = 0): self.statusBar().showMessage(message, timeout)
    def closeEvent(self, event): self.thread_pool.waitForDone(-1); event.accept()

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

