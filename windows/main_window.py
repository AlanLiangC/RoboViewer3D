import os
import socket
import time
import importlib

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QWidget, QGridLayout,\
                            QComboBox, QPushButton, QLabel
from PyQt5.QtCore import Qt

from .utils import common
from .utils.viewer_engine import gl, AL_viewer
from .utils.load_dataset import load_dataset_objectss
from config import dataset_configs
from roboprojects import projects_infos
class RoboMainWindow(QMainWindow):

    def __init__(self) -> None:
        super(RoboMainWindow, self).__init__()

        host_name = socket.gethostname()
        if host_name == 'Liang':
            self.monitor = QDesktopWidget().screenGeometry(1)
            self.monitor.setHeight(int(self.monitor.height()*0.5))
            self.monitor.setWidth(int(self.monitor.width()*0.5))
        else:
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(int(self.monitor.height()))
            self.monitor.setWidth(int(self.monitor.width()))

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.grid_dimensions = 10
        self.row_idx = 0
        self.columns_idx = 0
        self.columns = 7
        self.init_window()

    def update_rawcol(self, force=False):
        self.columns_idx += 1
        if force or self.columns_idx >= self.columns:
            self.row_idx += 1
            self.columns_idx = 0

    def init_window(self):
        # Initialize
        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()
        self.viewer = AL_viewer()
        self.grid = gl.GLGridItem()
        self.centerWidget.setLayout(self.layout)

        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.setCameraPosition(distance=2*self.grid_dimensions)
        self.layout.addWidget(self.viewer, self.row_idx, self.columns_idx, 1, self.columns)
        self.update_rawcol(force=True)

        # grid
        self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        self.grid.setSpacing(1, 1)
        self.grid.translate(0, 0, -2)
        self.viewer.addItem(self.grid)

        # ComboBox
        # select dataset
        self.select_dataset_cbox = QComboBox()
        self.layout.addWidget(self.select_dataset_cbox, self.row_idx, self.columns_idx)
        self.update_rawcol()
        supported_datasets = ['Nuscenes']
        self.select_dataset_cbox.addItems(supported_datasets)

        # ComboBox
        # select dataset
        self.select_split_cbox = QComboBox()
        self.layout.addWidget(self.select_split_cbox, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.select_split_cbox.addItems(['train', 'val'])

        # Button
        # load dataset
        self.load_dataset_button = QPushButton('Load Dataset')
        self.layout.addWidget(self.load_dataset_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.load_dataset_button.clicked.connect(self.load_dataset)

        # Button
        # prev view
        self.prev_view_button = QPushButton('<<<')
        self.layout.addWidget(self.prev_view_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.prev_view_button.clicked.connect(self.decrement_index)

        # Qlabel
        # show sample index
        self.sample_index_info = QLabel("")
        self.sample_index_info.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.sample_index_info, self.row_idx, self.columns_idx)
        self.update_rawcol()

        # Button
        # next view
        self.next_view_button = QPushButton('>>>')
        self.layout.addWidget(self.next_view_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.next_view_button.clicked.connect(self.increment_index)

        # Button
        # show gt seg 
        self.show_gt_seg_button = QPushButton('Show GT Seg')
        self.layout.addWidget(self.show_gt_seg_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.show_gt_seg_button.pressed.connect(self.show_points_mesh_w_seg_workspace)
        self.show_gt_seg_button.released.connect(self.show_current_mesh)

        # ComboBox
        # select project
        self.select_project_cbox = QComboBox()
        self.layout.addWidget(self.select_project_cbox, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.select_project_cbox.addItems(projects_infos.keys())

        # Button
        # open project window
        self.open_project_window_button = QPushButton('Load Project')
        self.layout.addWidget(self.open_project_window_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.open_project_window_button.clicked.connect(self.load_project)

    def reset_window(self):
        self.reset_viewer()
        self.sample_index_info.setText("")

    def reset_viewer(self):

        self.viewer.items = []
        # self.sample_index_info.setText("")

    def check_index_overflow(self) -> None:
        assert hasattr(self, 'data_list')
        totle_lenth = len(self.data_list)

        if self.index == -1:
            self.index = totle_lenth - 1

        if self.index >= totle_lenth:
            self.index = 0
    
    def decrement_index(self) -> None:

        self.index -= 1
        self.check_index_overflow()
        self.show_points_mesh_workspace()

    def increment_index(self) -> None:

        self.index += 1
        self.check_index_overflow()
        self.show_points_mesh_workspace()

    def load_dataset(self):
        dataset = self.select_dataset_cbox.currentText()
        dataset_split = self.select_split_cbox.currentText()
        dataset_config = dataset_configs[dataset]
        self.dataset_object = load_dataset_objectss[dataset](dataset_config)
        self.data_list = self.dataset_object.load_nuscenes(dataset_split)
        self.index = 0
        self.sample_index_info.setText(f"{self.index}")

        self.show_points_mesh_workspace()

    def mmdet2pointcept(self, data_dict):
        sample_dict = dict(
            lidar_path = os.path.join(self.dataset_object.data_root,
                                      'samples/LIDAR_TOP',
                                      data_dict['lidar_points']['lidar_path']),

            gt_segment_path = os.path.join(self.dataset_object.data_root,
                                      'lidarseg/v1.0-trainval',
                                      data_dict['pts_semantic_mask_path']),

            lidar_token = data_dict['token']
        )
        return sample_dict

    def show_points_mesh_workspace(self):
        self.reset_viewer()
        self.sample_index_info.setText(f"{self.index}")
        self.data_dict = self.dataset_object.analysis_data(self.data_list[self.index],
                                                      seg=True)
        mesh = common.get_points_mesh(self.data_dict['coord'], 1.5)
        self.current_mesh = mesh
        self.viewer.addItem(mesh)
        self.succecc_show = True

    def show_points_mesh_w_seg_workspace(self, segment=None):
        self.reset_viewer()
        if segment is None:
            segment = self.data_dict['segment']
        point_colors = self.dataset_object.color_map[segment+1]
        mesh = common.get_points_mesh(self.data_dict['coord'], 
                                      1.5, 
                                      colors=point_colors)
        self.viewer.addItem(mesh)

    def show_custom_points(self, points, size=1.5, colors=None):
        self.reset_viewer()
        mesh = common.get_points_mesh(points, 
                                      size=size, 
                                      colors=colors)
        self.viewer.addItem(mesh)


    def show_current_mesh(self):
        self.reset_viewer()
        self.viewer.addItem(self.current_mesh)

    def load_project(self):
        print('loading peoject window...')
        project_name = self.select_project_cbox.currentText()
        project_info = projects_infos[project_name]
        project_type, project_window_type = project_info['type'], project_info['window']
        project_module = importlib.import_module(f'roboprojects.{project_type}')
        project_config = project_module.projects_configs[project_name]
        if project_window_type == 'common':
            project_window = project_module.CommonWindow
        self.project_window = project_window(self)
        self.project_window.show_window()
        inference_func = project_module.get_inferencer
        setattr(self.project_window, 'inference_func', inference_func)
        setattr(self.project_window, 'project_config', project_config)
        print(f'loading peoject window {project_name} successful!')


