import os
import pickle
import numpy as np


class Nuscenes:

    COLOR_MAP = np.array(
        [
            [0,   0,   0, 255],  # 0 ignore
            [255, 158, 0, 255],  # 1 barrier  orange
            [0, 0, 230, 255],    # 2 bicycle  Blue
            [47, 79, 79, 255],   # 3 bus  Darkslategrey
            [220, 20, 60, 255],  # 4 car  Crimson
            [255, 69, 0, 255],   # 5 construction_vehicle  Orangered
            [255, 140, 0, 255],  # 6 motorcycle  Darkorange
            [233, 150, 70, 255], # 7 pedestrian  Darksalmon
            [255, 61, 99, 255],  # 8 traffic_cone  Red
            [112, 128, 144, 255],# 9 trailer  Slategrey
            [222, 184, 135, 255],# 10 truck Burlywood
            [0, 175, 0, 255],    # 11 driveable_surface  Green
            [165, 42, 42, 255],  # 12 other_flat  nuTonomy green
            [0, 207, 191, 255],  # 13 sidewalk, road, lane_marker, other_ground
            [75, 0, 75, 255], # 14 terrain, sidewalk
            [255, 0, 0, 255], # 15 manmade
            [0, 0, 128, 255], # 15 vegetation
        ])

    def __init__(self, dataset_config) -> None:
        self.ignore_index = -1
        self.dataset_config = dataset_config
        self.data_root = dataset_config['data_root']
        self.learning_map = self.get_learning_map(self.ignore_index)
        self.color_map = self.COLOR_MAP / 255

    def load_nuscenes(self, split):
        dataset_pkl = os.path.join(self.data_root, 
                                self.dataset_config['pkl_file'][split])
        f = open(dataset_pkl, 'rb')
        data_list = pickle.load(f)['data_list']
        print(f"{len(data_list)} samples had been loaded!")
        return data_list

    def analysis_data(self, sample_dict, bbox=False, seg=False):
        lidar_points_path = sample_dict['lidar_points']['lidar_path']
        lidar_path = os.path.join(self.data_root, 'samples/LIDAR_TOP/', lidar_points_path)
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        if seg:
            pts_semantic_mask_path = sample_dict['pts_semantic_mask_path']
            gt_segment_path = os.path.join(
                self.data_root, "lidarseg/v1.0-trainval", pts_semantic_mask_path)
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64)
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index

        if bbox:
            pass
        else:
            bbox = []

        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            bbox=bbox
        )
        return data_dict

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map




def load_semantickitti(dataset_config):
    pass


load_dataset_objectss = {
    'Nuscenes': Nuscenes,
    'SemanticKITTI': Nuscenes
}