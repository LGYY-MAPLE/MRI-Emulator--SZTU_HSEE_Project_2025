import datetime
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QSpinBox, QComboBox, QPushButton, QFileDialog,
                             QMessageBox, QTabWidget, QGroupBox, QFormLayout,
                             QFrame, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QAbstractItemView)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 样式表
class DarkStyle:
    """ 定义全程序的深色主题样式 (Dark Theme) """
    STYLESHEET = """
    QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; font-family: "Microsoft YaHei"; font-size: 10pt; }
    QLabel { color: #ffffff; font-weight: 500; }
    QGroupBox { border: 2px solid #505050; border-radius: 6px; margin-top: 12px; padding-top: 25px; font-weight: bold; color: #61dafb; background-color: #252526; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; background-color: #1e1e1e; }
    QPushButton { background-color: #3c3c3c; color: #fff; border: 1px solid #6e6e6e; border-radius: 4px; padding: 5px; }
    QPushButton:hover { background-color: #094771; border-color: #0078d4; }
    QLineEdit, QSpinBox, QComboBox { background-color: #3c3c3c; color: #fff; border: 1px solid #6e6e6e; border-radius: 3px; padding: 4px; }
    QSlider::groove:horizontal { border: 1px solid #3c3c3c; height: 6px; background: #2d2d2d; border-radius: 3px; }
    QSlider::handle:horizontal { background: #0078d4; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
    QTabWidget::pane { border: 1px solid #505050; }
    QTabBar::tab { background: #2d2d2d; color: #ccc; padding: 6px 15px; }
    QTabBar::tab:selected { background: #3c3c3c; color: #fff; border-bottom: 2px solid #0078d4; }
    QTableWidget { background-color: #252526; gridline-color: #505050; border: 1px solid #505050; }
    QHeaderView::section { background-color: #333333; color: #fff; padding: 4px; border: 1px solid #505050; }
    """


# 主窗口类
class MRISimulatorWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.ui_initialized = False
        self.model = model
        self.setWindowTitle("MRI 虚拟仿真工作站")
        self.resize(1400, 900)
        self.setStyleSheet(DarkStyle.STYLESHEET)

        # 交互状态
        self.scan_angle = 0.0
        self.is_dragging = False
        self.last_mouse_y = 0
        self.current_patient = {}  # {name, id, ...}

        # === 核心状态管理 ===
        # 记录三个视图当前的层数索引 (分别记忆)
        self.slice_indices = {
            'axial': 10,
            'sagittal': 64,
            'coronal': 64
        }
        # 当前激活的视图 (默认横断面)
        self.active_view = 'axial'
        default_params = {
            'seq_idx': 0, 'tr': 500, 'te': 20, 'ti': 0, 'fa': 90,
            'etl': 1, 'esp': 10, 'fov': 300, 'matrix_idx': 2,  # 256*256
            'thick': 1, 'gap': 0, 'angle': 0.0
        }
        self.view_params = {
            'axial': default_params.copy(),
            'sagittal': default_params.copy(),
            'coronal': default_params.copy()
        }
        self.init_ui()

        # 初始化数据加载
        self.load_patient_table()
        self.load_history_table()

        # 加载初始层数范围
        dims = self.model.get_dimensions()
        if dims:
            # 默认切在中间
            self.slice_indices['coronal'] = dims[0] // 2
            self.slice_indices['sagittal'] = dims[1] // 2
            self.slice_indices['axial'] = dims[2] // 2

        self.refresh_slider_for_active_view()
        self.load_params_to_ui(self.active_view)
        self.ui_initialized = True
        self.update_image()

    def init_ui(self):
        """ 初始化所有 UI 组件 """
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # === 1. 左侧控制面板 ===
        left_panel = QWidget();
        left_layout = QVBoxLayout(left_panel);
        left_panel.setFixedWidth(360)

        # 序列选择
        seq_group = QGroupBox("序列选择")
        self.combo_seq = QComboBox();
        self.combo_seq.addItems(["T1 加权", "T2 加权", "FLAIR", "PD 加权", "EPI", "FSE"]);
        self.combo_seq.currentIndexChanged.connect(self.on_sequence_change)
        sl = QFormLayout();
        sl.addRow("序列:", self.combo_seq);
        seq_group.setLayout(sl)

        # 物理参数控制
        param_group = QGroupBox("物理参数");
        p_layout = QFormLayout()

        self.slider_tr, self.spin_tr = self.create_param_pair(100, 10000, 500, " ms");
        p_layout.addRow("TR:", self.create_container(self.slider_tr, self.spin_tr))

        self.slider_te, self.spin_te = self.create_param_pair(1, 500, 20, " ms");
        p_layout.addRow("TE:", self.create_container(self.slider_te, self.spin_te))

        self.slider_ti, self.spin_ti = self.create_param_pair(100, 4000, 2000, " ms");
        self.ti_container = self.create_container(self.slider_ti, self.spin_ti);
        self.label_ti = QLabel("TI:");
        p_layout.addRow(self.label_ti, self.ti_container)

        self.slider_fa, self.spin_fa = self.create_param_pair(0, 180, 90, " °");
        p_layout.addRow("FA:", self.create_container(self.slider_fa, self.spin_fa))

        self.slider_etl, self.spin_etl = self.create_param_pair(1, 32, 8, "");
        self.etl_container = self.create_container(self.slider_etl, self.spin_etl);
        self.label_etl = QLabel("ETL:");
        p_layout.addRow(self.label_etl, self.etl_container)

        self.slider_esp, self.spin_esp = self.create_param_pair(1, 100, 10, " ms");
        self.esp_container = self.create_container(self.slider_esp, self.spin_esp);
        self.label_esp = QLabel("ESP:");
        p_layout.addRow(self.label_esp, self.esp_container)
        param_group.setLayout(p_layout)

        # 几何参数
        geo_group = QGroupBox("几何与切片");
        g_layout = QFormLayout()

        # 注意：这里移除了 combo_view (视图选择下拉框)，改为点击图像切换

        self.slider_fov, self.spin_fov = self.create_param_pair(100, 400, 300, " mm");
        g_layout.addRow("FOV:", self.create_container(self.slider_fov, self.spin_fov))
        self.combo_res = QComboBox()
        self.combo_res.addItems(["64 * 64", "128 * 128", "256 * 256", "512 * 512"])

        self.combo_res.setEditable(True)
        self.combo_res.lineEdit().setReadOnly(True)
        self.combo_res.lineEdit().setAlignment(Qt.AlignCenter)
        for i in range(self.combo_res.count()):
            self.combo_res.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)

        self.combo_res.setCurrentText("256 * 256")
        self.combo_res.currentIndexChanged.connect(self.update_image)
        g_layout.addRow("Matrix:", self.combo_res)

        self.slider_thick, self.spin_thick = self.create_param_pair(1, 20, 1, " mm");
        g_layout.addRow("Thick:", self.create_container(self.slider_thick, self.spin_thick))
        self.slider_gap, self.spin_gap = self.create_param_pair(0, 20, 0, " mm");
        g_layout.addRow("Gap:", self.create_container(self.slider_gap, self.spin_gap))

        # 滑块现在只控制“当前激活”的视图
        self.slider_slice = QSlider(Qt.Horizontal);
        # 注意：这里改了连接的函数
        self.slider_slice.valueChanged.connect(self.on_slice_slider_change);
        self.label_slice_info = QLabel("0/0")
        sb = QHBoxLayout();
        sb.addWidget(self.slider_slice);
        sb.addWidget(self.label_slice_info);
        g_layout.addRow("层位:", sb);
        g_layout.addRow(QLabel("提示: 点击右侧图像切换控制对象"))
        geo_group.setLayout(g_layout)

        # 功能按键
        btn_box = QVBoxLayout()
        btn_snap = QPushButton("📸 Scan");
        btn_snap.clicked.connect(self.handle_snapshot);
        btn_snap.setStyleSheet("background-color: #006600;")
        btn_box.addWidget(btn_snap)

        row1 = QHBoxLayout();
        b1 = QPushButton("导入模型");
        b1.clicked.connect(self.handle_load);
        row1.addWidget(b1)
        b2 = QPushButton("关于作者");
        b2.clicked.connect(self.show_author);
        row1.addWidget(b2)
        btn_box.addLayout(row1)

        row2 = QHBoxLayout();
        b3 = QPushButton("导出 DICOM");
        b3.clicked.connect(self.handle_export_dicom);
        row2.addWidget(b3)
        b4 = QPushButton("导出 K-Space");
        b4.clicked.connect(self.handle_export_kspace);
        row2.addWidget(b4)
        btn_box.addLayout(row2)

        left_layout.addWidget(seq_group);
        left_layout.addWidget(param_group);
        left_layout.addWidget(geo_group);
        left_layout.addLayout(btn_box);
        left_layout.addStretch()

        # === 2. 中间图像显示 (2x2 网格) ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # 使用一个 Figure，内含 4 个子图
        self.fig = Figure(facecolor='#202020');
        self.canvas = FigureCanvas(self.fig);

        # 创建 2x2 布局
        # gs[0,0]: Axial, gs[0,1]: Sagittal
        # gs[1,0]: Coronal, gs[1,1]: K-Space
        gs = self.fig.add_gridspec(2, 2, hspace=0.15, wspace=0.15, left=0.01, right=0.99, bottom=0.01, top=0.95)

        self.ax_axial = self.fig.add_subplot(gs[0, 0])
        self.ax_sag = self.fig.add_subplot(gs[0, 1])
        self.ax_cor = self.fig.add_subplot(gs[1, 0])
        self.ax_k = self.fig.add_subplot(gs[1, 1])

        # 隐藏所有坐标轴
        for ax in [self.ax_axial, self.ax_sag, self.ax_cor, self.ax_k]:
            ax.axis('off')

        # 绑定点击事件，用于切换“激活视图”
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        center_layout.addWidget(self.canvas)

        main_layout.addWidget(left_panel);
        main_layout.addWidget(center_panel, stretch=1)

        # === 3. 右侧功能选项卡 ===
        self.right_tab = QTabWidget();
        self.right_tab.setFixedWidth(320)

        # Scout (定位像)
        scout_w = QWidget();
        sl = QVBoxLayout(scout_w)
        self.scout_fig = Figure(facecolor='#202020');
        self.scout_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.scout_canvas = FigureCanvas(self.scout_fig);
        self.ax_scout = self.scout_fig.add_subplot(111);
        self.ax_scout.axis('off')
        sl.addWidget(QLabel("定位图 (仅横断面有效)"));
        sl.addWidget(self.scout_canvas)

        self.scout_canvas.mpl_connect('button_press_event', self.on_scout_press)
        self.scout_canvas.mpl_connect('motion_notify_event', self.on_scout_move)
        self.scout_canvas.mpl_connect('button_release_event', self.on_scout_release)
        self.scout_canvas.mpl_connect('scroll_event', self.on_scout_scroll)
        self.right_tab.addTab(scout_w, "定位")

        # Patients
        pat_w = QWidget();
        pl = QVBoxLayout(pat_w)
        self.pat_table = QTableWidget();
        self.pat_table.setColumnCount(3);
        self.pat_table.setHorizontalHeaderLabels(["ID", "姓名", "性别"])
        self.pat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pat_table.setSelectionBehavior(QAbstractItemView.SelectRows);
        self.pat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pat_table.itemClicked.connect(self.on_patient_select)

        form_box = QGroupBox("注册新病人");
        fl = QFormLayout()
        self.inp_name = QLineEdit();
        self.inp_id = QLineEdit();
        self.inp_age = QSpinBox();
        self.inp_age.setRange(0, 120);
        self.inp_sex = QComboBox();
        self.inp_sex.addItems(["男", "女", "其他"])
        fl.addRow("姓名:", self.inp_name);
        fl.addRow("ID:", self.inp_id);
        fl.addRow("年龄:", self.inp_age);
        fl.addRow("性别:", self.inp_sex)
        btn_reg = QPushButton("注册");
        btn_reg.clicked.connect(self.handle_register)
        btn_del = QPushButton("删除选中");
        btn_del.clicked.connect(self.handle_delete_patient)
        form_box.setLayout(fl)

        pl.addWidget(self.pat_table);
        pl.addWidget(form_box);
        pl.addWidget(btn_reg);
        pl.addWidget(btn_del)
        self.right_tab.addTab(pat_w, "病人")

        # History
        hist_w = QWidget();
        hl = QVBoxLayout(hist_w)
        h_top = QHBoxLayout()
        self.lbl_hist_status = QLabel("当前显示: 全部记录")
        self.lbl_hist_status.setStyleSheet("color: #aaa;")
        btn_show_all = QPushButton("显示全部");
        btn_show_all.setFixedWidth(80)
        btn_show_all.clicked.connect(self.handle_show_all_history)
        h_top.addWidget(self.lbl_hist_status);
        h_top.addWidget(btn_show_all)

        self.hist_table = QTableWidget();
        self.hist_table.setColumnCount(3);
        self.hist_table.setHorizontalHeaderLabels(["时间", "病人", "序列"])
        self.hist_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.hist_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.hist_table.itemClicked.connect(self.on_history_select)

        btn_del_hist = QPushButton("删除选中记录");
        btn_del_hist.clicked.connect(self.handle_delete_history)

        hl.addLayout(h_top);
        hl.addWidget(self.hist_table);
        hl.addWidget(btn_del_hist)
        self.right_tab.addTab(hist_w, "历史")

        main_layout.addWidget(self.right_tab)
        self.on_sequence_change(0)

    # === 辅助控件创建 ===
    def create_param_pair(self, min_v, max_v, val, suffix):
        slider = QSlider(Qt.Horizontal);
        slider.setRange(min_v, max_v);
        slider.setValue(val)
        spin = QSpinBox();
        spin.setRange(min_v, max_v);
        spin.setValue(val);
        spin.setSuffix(suffix);
        spin.setAlignment(Qt.AlignCenter)
        slider.valueChanged.connect(spin.setValue);
        spin.valueChanged.connect(slider.setValue)
        # 实时更新图像
        slider.valueChanged.connect(self.update_image)
        return slider, spin

    def create_container(self, w1, w2):
        w = QWidget();
        l = QHBoxLayout(w);
        l.setContentsMargins(0, 0, 0, 0)
        w1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed);
        w2.setFixedWidth(100)
        l.addWidget(w1);
        l.addWidget(w2);
        return w

    def refresh_slice_range(self):
        """ 废弃旧的单一刷新逻辑，改为 refresh_slider_for_active_view """
        pass

    def on_sequence_change(self, idx):
        presets = [
            (500, 20, 90, 0, 0, 0),  # T1
            (4000, 100, 90, 0, 0, 0),  # T2
            (9000, 120, 90, 1, 0, 0),  # FLAIR
            (2000, 20, 90, 0, 0, 0),  # PD
            (100, 30, 20, 0, 0, 1),  # EPI
            (3000, 80, 90, 0, 1, 1)  # FSE
        ]
        tr, te, fa, ti_v, etl_v, esp_v = presets[idx]
        self.spin_tr.setValue(tr);
        self.spin_te.setValue(te);
        self.spin_fa.setValue(fa)

        self.label_ti.setVisible(ti_v);
        self.ti_container.setVisible(ti_v)
        self.label_etl.setVisible(etl_v);
        self.etl_container.setVisible(etl_v)
        self.label_esp.setVisible(esp_v);
        self.esp_container.setVisible(esp_v)
        self.update_image()

    # === 核心交互：视图切换 ===
    def on_canvas_click(self, event):
        """ 检测用户点击，切换激活视图，并切换参数上下文 """
        new_view = None
        if event.inaxes == self.ax_axial: new_view = 'axial'
        elif event.inaxes == self.ax_sag: new_view = 'sagittal'
        elif event.inaxes == self.ax_cor: new_view = 'coronal'

        if new_view and new_view != self.active_view:
            # 1. 保存当前 UI 参数到旧视图的记录中
            self.view_params[self.active_view] = self.get_current_ui_params()

            # 2. 切换视图
            self.active_view = new_view

            # 3. 加载新视图的参数到 UI
            self.load_params_to_ui(new_view)

            # 4. 刷新图像
            self.update_image()

    def refresh_slider_for_active_view(self):
        """ 根据当前激活的视图，更新滑块的范围和当前值 """
        self.slider_slice.blockSignals(True)

        # 获取最大层数
        max_slices = 0
        if hasattr(self.model, 'get_max_slice'):
            max_slices = self.model.get_max_slice(self.active_view)
        else:
            # Fallback
            dims = self.model.get_dimensions()
            if dims: max_slices = dims[2]

        # 设置范围
        if max_slices > 0:
            self.slider_slice.setRange(0, max_slices - 1)
        else:
            self.slider_slice.setRange(0, 0)

        # 设置当前值
        current_idx = self.slice_indices.get(self.active_view, 0)
        if max_slices > 0 and current_idx >= max_slices:
            current_idx = max_slices - 1

        self.slider_slice.setValue(current_idx)
        self.label_slice_info.setText(f"{current_idx + 1}/{max_slices}")

        self.slider_slice.blockSignals(False)

    def on_slice_slider_change(self):
        """ 滑块拖动时，只更新当前激活视图的层数索引 """
        val = self.slider_slice.value()
        self.slice_indices[self.active_view] = val

        # 更新 Label
        max_s = 0
        if hasattr(self.model, 'get_max_slice'):
            max_s = self.model.get_max_slice(self.active_view)
        self.label_slice_info.setText(f"{val + 1}/{max_s}")

        self.update_image()

    def get_current_ui_params(self):
        """ 从 UI 控件获取当前的所有参数值 """
        return {
            'seq_idx': self.combo_seq.currentIndex(),
            'tr': self.spin_tr.value(),
            'te': self.spin_te.value(),
            'ti': self.spin_ti.value(),
            'fa': self.spin_fa.value(),
            'etl': self.spin_etl.value(),
            'esp': self.spin_esp.value(),
            'fov': self.spin_fov.value(),
            'matrix_idx': self.combo_res.currentIndex(),
            'thick': self.spin_thick.value(),
            'gap': self.spin_gap.value(),
            # 扫描角度现在也归属到具体视图
            'angle': self.scan_angle
        }

    def load_params_to_ui(self, view_name):
        """ 将指定视图的参数加载到 UI 控件中，并暂时屏蔽信号 """
        p = self.view_params.get(view_name)
        if not p: return

        # 屏蔽所有控件的信号，防止在赋值时触发 update_image
        widgets = [self.combo_seq, self.slider_tr, self.spin_tr, self.slider_te, self.spin_te,
                   self.slider_ti, self.spin_ti, self.slider_fa, self.spin_fa,
                   self.slider_etl, self.spin_etl, self.slider_esp, self.spin_esp,
                   self.slider_fov, self.spin_fov, self.combo_res,
                   self.slider_thick, self.spin_thick, self.slider_gap, self.spin_gap]

        for w in widgets: w.blockSignals(True)

        # 赋值
        self.combo_seq.setCurrentIndex(p['seq_idx'])
        self.on_sequence_change(p['seq_idx'])  # 手动更新一下显隐状态(TI/ETL等)

        self.spin_tr.setValue(p['tr'])
        self.spin_te.setValue(p['te'])
        self.spin_ti.setValue(p['ti'])
        self.spin_fa.setValue(p['fa'])
        self.spin_etl.setValue(p['etl'])
        self.spin_esp.setValue(p['esp'])
        self.spin_fov.setValue(p['fov'])
        self.combo_res.setCurrentIndex(p['matrix_idx'])
        self.spin_thick.setValue(p['thick'])
        self.spin_gap.setValue(p['gap'])

        # 恢复全局旋转变量
        self.scan_angle = p['angle']

        # 恢复信号
        for w in widgets: w.blockSignals(False)

        # 刷新滑块范围 (层数)
        self.refresh_slider_for_active_view()

    # === 绘图渲染逻辑 (4分屏) ===
    def update_image(self):
        if self.ui_initialized:
            self.view_params[self.active_view] = self.get_current_ui_params()
        # 定义视图配置
        views_config = [
            ('axial', self.ax_axial, "横断面 (Axial)"),
            ('sagittal', self.ax_sag, "矢状面 (Sagittal)"),
            ('coronal', self.ax_cor, "冠状面 (Coronal)")
        ]

        # === 独立渲染 ===
        for plane_name, ax, title in views_config:
            # 获取该平面的独立参数字典
            p = self.view_params[plane_name]

            # 解析参数
            # 序列名称需通过 index 从列表反查，或者我们在 dict 里直接存了 index
            seq_name = self.combo_seq.itemText(p['seq_idx'])

            # 解析 Matrix (例如 "256 * 256")
            mat_txt = self.combo_res.itemText(p['matrix_idx'])
            matrix_size = int(mat_txt.split('*')[0].strip()) if '*' in mat_txt else int(mat_txt)

            # 获取层数索引
            idx = self.slice_indices.get(plane_name, 0)
            max_s = self.model.get_max_slice(plane_name) if hasattr(self.model, 'get_max_slice') else 0
            if idx >= max_s and max_s > 0: idx = max_s - 1

            # 计算图像 (使用 p[...] 中的参数，互不干扰)
            img = self.model.calculate_image(
                seq_type=seq_name,
                tr=p['tr'], te=p['te'], ti=p['ti'], fa=p['fa'],
                slice_idx=idx,
                etl=p['etl'], esp=p['esp'],
                fov=p['fov'], thickness=p['thick'],
                matrix_size=matrix_size,
                rotation=p['angle'],  # 每个视图使用自己的角度
                view_plane=plane_name
            )

            ax.clear()
            if img is not None:
                disp = np.rot90(img)
                p99 = np.percentile(disp, 99) if disp.max() > 0 else 1
                ax.imshow(disp, cmap='gray', vmin=0, vmax=p99,
                          extent=[-p['fov'] / 2, p['fov'] / 2, -p['fov'] / 2, p['fov'] / 2])

                # 标题高亮逻辑
                color = '#FFFF00' if plane_name == self.active_view else '#FFFFFF'
                weight = 'bold' if plane_name == self.active_view else 'normal'
                ax.set_title(f"{title} [{idx + 1}/{max_s}]", color=color, fontsize=10, fontweight=weight)
            ax.axis('off')

        # 绘制 K 空间 (使用当前激活视图的参数)
        # 注意：这里直接取 active_view 的参数 p_active
        p_active = self.view_params[self.active_view]
        seq_active = self.combo_seq.itemText(p_active['seq_idx'])
        mat_active_txt = self.combo_res.itemText(p_active['matrix_idx'])
        mat_active_size = int(mat_active_txt.split('*')[0]) if '*' in mat_active_txt else 256
        idx_active = self.slice_indices.get(self.active_view, 0)

        self.model.calculate_image(
            seq_active, p_active['tr'], p_active['te'], p_active['ti'], p_active['fa'],
            idx_active, p_active['etl'], p_active['esp'], p_active['fov'], p_active['thick'],
            0, mat_active_size, p_active['angle'], self.active_view
        )

        self.ax_k.clear()
        if self.model.k_space is not None:
            k_disp = np.rot90(np.log(np.abs(self.model.k_space) + 1))
            self.ax_k.imshow(k_disp, cmap='gray')
            self.ax_k.set_title(f"K-Space ({self.active_view})", color='#AAAAAA', fontsize=9)
        self.ax_k.axis('off')

        self.canvas.draw()

        # 4. 更新定位像 (仅当激活视图为 Axial 时，且使用 Axial 的参数)
        if self.active_view == 'axial':
            dims = self.model.get_dimensions()
            # 这里的 thick 和 fov 必须取 Axial 自己的参数
            p_ax = self.view_params['axial']
            self.update_scout(self.slice_indices['axial'], p_ax['thick'], p_ax['fov'], dims)
        else:
            self.ax_scout.clear()
            self.ax_scout.text(0.5, 0.5, "定位功能仅支持横断面", color='#888888', ha='center', va='center', fontsize=10)
            self.ax_scout.set_facecolor('black')
            self.ax_scout.axis('off')
            self.scout_canvas.draw()

    # === 交互 (Scout) ===
    def on_scout_press(self, event):
        if self.active_view != 'axial': return
        if event.inaxes == self.ax_scout and event.button == 1: self.is_dragging = True; self.last_mouse_y = event.ydata

    def on_scout_move(self, event):
        if self.is_dragging and event.inaxes == self.ax_scout:
            dy = event.ydata - self.last_mouse_y;
            self.last_mouse_y = event.ydata;
            self.slider_slice.setValue(int(self.slider_slice.value() - dy * 1.5))

    def on_scout_scroll(self, event):
        if event.inaxes == self.ax_scout:
            self.scan_angle += 5.0 if event.button == 'up' else -5.0;
            if self.active_view == 'axial':
                self.view_params['axial']['angle'] = self.scan_angle

            self.update_image()

    def on_scout_release(self, event):
        self.is_dragging = False

    def update_scout(self, idx, thick, fov, dims):
        self.ax_scout.clear()
        if self.model.t1_vol is not None:
            mid = dims[1] // 2
            sag = np.flipud(np.rot90(self.model.t1_vol[:, mid, :]))
            h, w = sag.shape
            self.ax_scout.imshow(sag, cmap='gray', extent=[0, w, 0, h])
            self.ax_scout.set_xlim(0, w);
            self.ax_scout.set_ylim(0, h)

            cx, cy = w / 2, h - idx
            diag = np.sqrt(w ** 2 + h ** 2) * 1.2
            t = transforms.Affine2D().rotate_deg_around(cx, cy, -self.scan_angle) + self.ax_scout.transData

            rect = patches.Rectangle((cx - diag / 2, cy - thick / 2), diag, thick, linewidth=0, color='red', alpha=0.5)
            rect.set_transform(t);
            self.ax_scout.add_patch(rect)

            if fov / 300.0 < 1.0:
                fh = h * (fov / 300.0)
                fr = patches.Rectangle((cx - w / 2, cy - fh / 2), w, fh, linewidth=1, edgecolor='yellow',
                                       facecolor='none', linestyle='--')
                fr.set_transform(t);
                self.ax_scout.add_patch(fr)
        self.ax_scout.axis('off');
        self.scout_canvas.draw()

    # === 病人/历史/导入/导出 (保持功能) ===
    def load_patient_table(self):
        self.pat_table.setRowCount(0)
        for p in self.model.patient_mgr.data:
            r = self.pat_table.rowCount();
            self.pat_table.insertRow(r)
            self.pat_table.setItem(r, 0, QTableWidgetItem(p['id']));
            self.pat_table.setItem(r, 1, QTableWidgetItem(p['name']));
            self.pat_table.setItem(r, 2, QTableWidgetItem(p['sex']))

    def handle_register(self):
        if not self.inp_name.text() or not self.inp_id.text(): QMessageBox.warning(self, "Warn",
                                                                                   "ID和姓名不能为空"); return
        info = {'name': self.inp_name.text(), 'id': self.inp_id.text(), 'age': self.inp_age.value(),
                'sex': self.inp_sex.currentText()}
        self.model.patient_mgr.add_patient(info);
        self.load_patient_table();
        QMessageBox.information(self, "OK", "注册成功")

    def handle_delete_patient(self):
        r = self.pat_table.currentRow()
        if r >= 0:
            deleted_id = self.model.patient_mgr.data[r]['id']
            self.model.patient_mgr.delete_patient(r)
            self.load_patient_table()
            if self.current_patient.get('id') == deleted_id: self.current_patient = {}; self.load_history_table()

    def on_patient_select(self):
        r = self.pat_table.currentRow()
        if r >= 0:
            self.current_patient = self.model.patient_mgr.data[r]
            self.load_history_table()
            self.lbl_hist_status.setText(f"当前显示: {self.current_patient['name']}")

    def load_history_table(self):
        self.hist_table.setRowCount(0)
        filter_id = self.current_patient.get('id')
        if filter_id:
            self.lbl_hist_status.setText(f"当前显示: {self.current_patient.get('name')}")
        else:
            self.lbl_hist_status.setText("当前显示: 全部记录")
        for i, h in enumerate(self.model.history_mgr.data):
            if filter_id and h.get('patient_id') != filter_id: continue
            r = self.hist_table.rowCount();
            self.hist_table.insertRow(r)
            item_time = QTableWidgetItem(h['time']);
            item_time.setData(Qt.UserRole, i)
            self.hist_table.setItem(r, 0, item_time);
            self.hist_table.setItem(r, 1, QTableWidgetItem(h['patient_name']));
            self.hist_table.setItem(r, 2, QTableWidgetItem(h['seq']))

    def handle_show_all_history(self):
        self.current_patient = {};
        self.pat_table.clearSelection();
        self.load_history_table()

    def handle_snapshot(self):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        params = {'seq': self.combo_seq.currentText(), 'tr': self.spin_tr.value(), 'te': self.spin_te.value(),
                  'ti': self.spin_ti.value(), 'fa': self.spin_fa.value(),
                  'etl': self.spin_etl.value(), 'esp': self.spin_esp.value(), 'fov': self.spin_fov.value(),
                  'thick': self.spin_thick.value(), 'gap': self.spin_gap.value(),
                  'slice': self.slider_slice.value(), 'matrix': self.combo_res.currentText(), 'angle': self.scan_angle}
        record = {'time': now, 'patient_name': self.current_patient.get('name', 'Anonymous'),
                  'patient_id': self.current_patient.get('id', '0000'), 'seq': params['seq'], 'params': params}
        self.model.history_mgr.add_record(record);
        self.load_history_table();
        QMessageBox.information(self, "Snapshot", "已记录所有参数")

    def on_history_select(self):
        r = self.hist_table.currentRow()
        if r < 0: return
        idx = self.hist_table.item(r, 0).data(Qt.UserRole)
        p = self.model.history_mgr.data[idx]['params']
        self.blockSignals(True)
        self.combo_seq.setCurrentText(p['seq']);
        self.on_sequence_change(self.combo_seq.currentIndex())
        self.spin_tr.setValue(p['tr']);
        self.slider_tr.setValue(p['tr'])
        self.spin_te.setValue(p['te']);
        self.slider_te.setValue(p['te'])
        self.spin_ti.setValue(p['ti']);
        self.slider_ti.setValue(p['ti'])
        self.spin_fa.setValue(p['fa']);
        self.slider_fa.setValue(p['fa'])
        self.spin_etl.setValue(p['etl']);
        self.slider_etl.setValue(p['etl'])
        self.spin_esp.setValue(p['esp']);
        self.slider_esp.setValue(p['esp'])
        self.spin_fov.setValue(p['fov']);
        self.slider_fov.setValue(p['fov'])
        self.spin_thick.setValue(p['thick']);
        self.slider_thick.setValue(p['thick'])
        self.spin_gap.setValue(p['gap']);
        self.slider_gap.setValue(p['gap'])
        # 还原层位: 更新当前激活视图的索引
        self.slice_indices[self.active_view] = p['slice']
        self.combo_res.setCurrentText(p['matrix'])
        self.scan_angle = p['angle']
        self.blockSignals(False)
        self.refresh_slider_for_active_view()
        self.update_image()

    def handle_delete_history(self):
        r = self.hist_table.currentRow()
        if r >= 0:
            idx = self.hist_table.item(r, 0).data(Qt.UserRole)
            self.model.history_mgr.delete_record(idx)
            self.load_history_table()

    def handle_load(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Open', '', 'MAT (*.mat)')
        if f and self.model.load_mat_file(f)[0]:
            # 重置索引到中间
            dims = self.model.get_dimensions()
            self.slice_indices = {'axial': dims[2] // 2, 'sagittal': dims[1] // 2, 'coronal': dims[0] // 2}
            self.refresh_slider_for_active_view()
            self.update_image()
            QMessageBox.information(self, "导入成功", "模型已加载")

    def handle_export_dicom(self):
        try:
            path, _ = QFileDialog.getSaveFileName(self, "保存 DICOM", "img.dcm", "DICOM (*.dcm)")
            if not path: return
            p_info = self.current_patient if self.current_patient else {'name': self.inp_name.text() or 'Anonymous',
                                                                        'id': self.inp_id.text() or '0000',
                                                                        'age': self.inp_age.value(),
                                                                        'sex': self.inp_sex.currentText()}
            # 导出当前激活视图
            self.model.calculate_image(self.combo_seq.currentText(), self.spin_tr.value(), self.spin_te.value(),
                                       self.spin_ti.value(), self.spin_fa.value(),
                                       self.slice_indices[self.active_view], view_plane=self.active_view)
            success, msg = self.model.export_dicom(path, p_info, self.combo_seq.currentText())
            if success:
                QMessageBox.information(self, "成功", f"DICOM 已保存至:\n{path}")
            else:
                QMessageBox.critical(self, "导出失败", str(msg))
        except Exception as e:
            QMessageBox.critical(self, "程序错误", str(e))

    def handle_export_kspace(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存 K-Space", "k.mat", "MAT (*.mat)")
        if path:
            # 确保 K 空间是当前视图的
            self.model.calculate_image(self.combo_seq.currentText(), self.spin_tr.value(), self.spin_te.value(),
                                       self.spin_ti.value(), self.spin_fa.value(),
                                       self.slice_indices[self.active_view], view_plane=self.active_view)
            if self.model.export_kspace(path): QMessageBox.information(self, "OK", "K-Space 导出成功")

    def show_author(self):
        content = """
        <h3 style='color: #0078d4;'>MRI 虚拟仿真工作站</h3>
        <p><b>医学成像原理及技术 - 课程设计小组</b></p>
        <hr>
        <p><b>👨‍💻 小组成员：</b></p>
        <ul style='line-height: 150%;'>
            <li><b>许威翔</b> (学号: 202300502116)
            <li><b>黄文华</b> (学号: 202300502118)
            <li><b>赵彦博</b> (学号: 202300502022)
        </ul>
        """
        QMessageBox.about(self, "关于作者", content)