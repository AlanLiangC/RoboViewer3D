import torch
import socket
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QWidget, QGridLayout,\
                            QComboBox, QPushButton, QLabel
from PyQt5.QtCore import Qt


class CommonWindow(QMainWindow):

    def __init__(self, main_window: QMainWindow) -> None:
        super(CommonWindow, self).__init__()

        host_name = socket.gethostname()
        if host_name == 'Liang':
            self.monitor = QDesktopWidget().screenGeometry(1)
            self.monitor.setHeight(int(self.monitor.height()*0.2))
            self.monitor.setWidth(int(self.monitor.width()*0.2))
        else:
            self.monitor = QDesktopWidget().screenGeometry(0)
            self.monitor.setHeight(int(self.monitor.height()))
            self.monitor.setWidth(int(self.monitor.width()))

        self.setGeometry(self.monitor)
        self.setAcceptDrops(True)

        self.main_window = main_window
        self.row_idx = 0
        self.columns_idx = 0
        self.columns = 7
        self.init_window()

    def update_rawcol(self, force=False):
        self.columns_idx += 1
        if force or self.columns_idx >= self.columns:
            self.row_idx += 1
            self.columns_idx = 0

    def show_window(self):
        self.show()

    def init_window(self):
        # Initialize
        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        self.layout = QGridLayout()
        self.centerWidget.setLayout(self.layout)

        # Button
        # load inferencer
        self.load_inferencer_button = QPushButton('Load Inferencer')
        self.layout.addWidget(self.load_inferencer_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.load_inferencer_button.clicked.connect(self.load_inferencer)

        # Button
        # inference
        self.inference_button = QPushButton('Inferencer')
        self.layout.addWidget(self.inference_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.inference_button.clicked.connect(self.inference)
        self.inference_button.setEnabled(False)

        # Button
        # show pred seg 
        self.show_pred_seg_button = QPushButton('Show Pred Seg')
        self.layout.addWidget(self.show_pred_seg_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.show_pred_seg_button.pressed.connect(self.show_pred_seg)
        self.show_pred_seg_button.released.connect(self.main_window.show_current_mesh)

        # Button 
        # show uncentainess by Evidence Theory
        self.show_uncentainess_seg_button = QPushButton('Show Uncentainess Seg')
        self.layout.addWidget(self.show_uncentainess_seg_button, self.row_idx, self.columns_idx)
        self.update_rawcol()
        self.show_uncentainess_seg_button.pressed.connect(self.show_uncertainess_seg)
        self.show_uncentainess_seg_button.released.connect(self.main_window.show_current_mesh)

    def load_inferencer(self):
        print('load inferencer...')
        self.inferencer = self.inference_func(self.project_config)
        print('load inferencer succussfull!')
        self.inference_button.setEnabled(True)

    def inference(self):
        data_dict = self.main_window.data_list[self.main_window.index]
        sample_dict = self.main_window.mmdet2pointcept(data_dict)
        self.inferencer.test_loader.dataset.data_list = [sample_dict]
        self.pred, self.pred_scores = self.inferencer.inference()

    def show_pred_seg(self):
        self.main_window.show_points_mesh_w_seg_workspace(segment=self.pred)

    def show_uncertainess_seg(self):
        '''
        Show pred uncertainess with evidence theory
        '''
        pred = torch.from_numpy(self.pred)
        pred_scores_softmax = torch.from_numpy(self.pred_scores) # [27815, 16]
        P = torch.gather(pred_scores_softmax, 1, pred.unsqueeze(1)).squeeze()
        R = torch.sum(pred_scores_softmax, dim=-1) - P
        P = P - 30
        P = torch.clamp_min(P, 0)
        p_fn, p_fp = 0.8 * torch.ones(1), 0.8 * torch.ones(1)
        m_p = torch.pow(p_fn, R) * (1 - torch.pow(p_fp, P))
        m_o = torch.pow(p_fp, P) * (1 - torch.pow(p_fn, R))

        uncentainess = 1 - m_p
        # uncentainess = torch.clamp_min(uncentainess, 0)
        
        colors = torch.ones([uncentainess.shape[0], 4])*255
        colors[:,3] = uncentainess

        self.main_window.show_custom_points(
            points = self.main_window.data_dict['coord'],
            size = 5,
            colors = colors
        )
