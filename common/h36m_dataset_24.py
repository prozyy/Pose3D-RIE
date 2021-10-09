# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy,json
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates, world_to_camera
import time,os
import pickle

h36m_skeleton = Skeleton(parents=[-1,  0,  0,  1,  2,  6,  5,  5,  6,  7,  8,  5, 6, 11, 12, 13, 14, 15, 16, 15, 16, 0, 9, 10],
       joints_left=[1,3,5,7,9,11,13,15,17,19,22],
       joints_right=[2,4,6,8,10,12,14,16,18,20,23])

class Human36mDataset(MocapDataset):
    def __init__(self, path3d, path2d, remove_static_joints=True, chunk_length=1, augment=False, pad=121, causal_shift=0, is_train=False):
        super().__init__(fps=50, skeleton=copy.deepcopy(h36m_skeleton))
        self.chunk_length = chunk_length
        self.augment = augment
        self.pad = pad
        self.causal_shift = causal_shift
        self.is_train = is_train
        self.train_subjects = ["S1", "S2", "S5", "S6", "S7", "S8", "setting1", "setting101", "setting102", "setting2", "setting4","setting6", "setting71", "setting72", "setting81", "setting82", "setting91", "setting92"]
        self.test_subjects = ["S9", "S11", "setting3", "setting5"]
        cam_params = json.load(open("data/cam_all_z+.json"))
        self._cameras = copy.deepcopy(cam_params)
        for sub,cameras in self._cameras.items():
            for i, cam in enumerate(cameras):
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')

                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000  # mm to meters

                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'], cam['center'], cam['radial_distortion'], cam['tangential_distortion']))

        # Load serialized dataset
        self._data_3d = np.load(path3d, allow_pickle=True)['positions_3d'].item()

        # if remove_static_joints:
        #     # Bring the skeleton to 17 joints instead of the original 32
        #     self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

        #     # Rewire shoulders to the correct parents
        #     self._skeleton._parents[11] = 8
        #     self._skeleton._parents[14] = 8

        data2d_file = np.load(path2d, allow_pickle=True)
        self._data_2d = data2d_file['positions_2d'].item()
        keypoints_symmetry = data2d_file['metadata'].item()['keypoints_symmetry']
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])

        if os.path.exists(path3d.replace(".npz",".pkl")):
            with open(path3d.replace(".npz",".pkl"), 'rb') as handle:
                self._data_3d = pickle.load(handle)
        else:
            self.process3DPose()
            with open(path3d.replace(".npz",".pkl"), 'wb') as handle:
                pickle.dump(self._data_3d,handle)

        self.normalize2DPose()
        self.getPairs()

        self.joints_left = self.skeleton().joints_left()
        self.joints_right = self.skeleton().joints_right()

    def supports_semi_supervised(self):
        return True

    def process3DPose(self):
        data_3d = self._data_3d
        self._data_3d = {}
        for subject, actions in data_3d.items():
            self._data_3d[subject] = {}
            for action_name, positions in actions.items():
                positions = positions[:, self.kept_joints]
                positions_3d = []
                for cam in self._cameras[subject]:
                    pos_3d = world_to_camera(positions, R=cam['orientation'], t=cam['translation'])
                    # pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                    if len(pos_3d.shape) == 4:
                        pos_3d = pos_3d.reshape(pos_3d.shape[0] * pos_3d.shape[1],pos_3d.shape[2],pos_3d.shape[3])
                    pos_3d[:, :] -= np.mean(pos_3d[:, 11:13],axis = 1,keepdims=True)  # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                self._data_3d[subject][action_name] = {'positions': positions, 'cameras': self._cameras[subject], 'positions_3d': positions_3d}

    def normalize2DPose(self):
        for subject in self._data_3d.keys():
            assert subject in self._data_2d, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in self._data_3d[subject].keys():
                assert action in self._data_2d[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
                if 'positions_3d' not in self._data_3d[subject][action]:
                    continue
                for cam_idx in range(len(self._data_2d[subject][action])):
                    # We check for >= instead of == because some videos in H3.6M contain extra frames
                    mocap_length = self._data_3d[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert self._data_2d[subject][action][cam_idx].shape[0] >= mocap_length

                    if self._data_2d[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        self._data_2d[subject][action][cam_idx] = self._data_2d[subject][action][cam_idx][:mocap_length]

                    cam = self._cameras[subject][cam_idx]
                    kps = self._data_2d[subject][action][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    self._data_2d[subject][action][cam_idx] = kps
                assert len(self._data_2d[subject][action]) == len(self._data_3d[subject][action]['positions_3d'])

    def getPairs(self):
        if self.is_train:
            subjects = self.train_subjects
        else:
            subjects = self.test_subjects

        for subject in subjects:
            assert subject in self._data_2d, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in self._data_3d[subject].keys():
                pose3dlist = self._data_3d[subject][action]["positions_3d"]
                pose2dlist = self._data_2d[subject][action]
                assert len(pose3dlist) == len(pose2dlist)
                for i in range(len(pose3dlist)):
                    assert pose3dlist[i].shape[0] == pose2dlist[i].shape[0]
                    n_chunks = (pose2dlist[i].shape[0] + self.chunk_length - 1) // self.chunk_length
                    offset = (n_chunks * self.chunk_length - pose2dlist[i].shape[0]) // 2
                    bounds = np.arange(n_chunks + 1) * self.chunk_length - offset
                    augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                    self._pairs += zip(["{:s}_{:s}_{:d}".format(subject, action, i)] * len(bounds - 1), bounds[:-1], bounds[1:], augment_vector)
                    if self.augment:
                        self._pairs += zip(["{:s}_{:s}_{:d}".format(subject, action, i)] * len(bounds - 1), bounds[:-1], bounds[1:], ~augment_vector)

    def getDataFromPair(self, index):

        subject_action_camIndex, start_3d, end_3d, flip = self._pairs[index]
        str_split_list = subject_action_camIndex.split("_")
        if len(str_split_list) == 3:
            subject, action, camIndex = str_split_list
        elif len(str_split_list) >= 4:
            subject = str_split_list[0]
            camIndex = str_split_list[-1]
            action = "_".join(str_split_list[1:-1])
        # print("subject:",subject," action:",action," camIndex:",camIndex)

        seq_2d = self._data_2d[subject][action][int(camIndex)]

        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            group_2d = np.pad(copy.deepcopy(seq_2d[low_2d:high_2d]), ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            group_2d = copy.deepcopy(seq_2d[low_2d:high_2d])

        if flip:
            # Flip 2D keypoints
            group_2d[:, :, 0] *= -1
            group_2d[:, self.kps_left + self.kps_right] = group_2d[:, self.kps_right + self.kps_left]

        seq_3d = self._data_3d[subject][action]["positions_3d"][int(camIndex)]

        low_3d = max(start_3d, 0)
        high_3d = min(end_3d, seq_3d.shape[0])
        pad_left_3d = low_3d - start_3d
        pad_right_3d = end_3d - high_3d
        if pad_left_3d != 0 or pad_right_3d != 0:
            group_3d = np.pad(copy.deepcopy(seq_3d[low_3d:high_3d]), ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
        else:
            group_3d = copy.deepcopy(seq_3d[low_3d:high_3d])

        if flip:
            # Flip 3D joints
            group_3d[:, :, 0] *= -1
            group_3d[:, self.joints_left + self.joints_right] = group_3d[:, self.joints_right + self.joints_left]

        group_cam = self._cameras[subject][int(camIndex)]['intrinsic']

        if flip:
            # Flip horizontal distortion coefficients
            group_cam[2] *= -1
            group_cam[7] *= -1

        return subject_action_camIndex, group_cam, group_3d, group_2d
