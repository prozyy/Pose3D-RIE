# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates, world_to_camera

h36m_skeleton = Skeleton(parents=[-1,  0,  0,  1,  2,  6,  5,  5,  6,  7,  8,  5, 6, 11, 12, 13, 14, 15, 16, 15, 16, 0, 9, 10],
       joints_left=[1,3,5,7,9,11,13,15,17,19,22],
       joints_right=[2,4,6,8,10,12,14,16,18,20,23])

h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70,  # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70,  # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110,  # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110,  # Only used for visualization
    },
]

h36m_goProCameras_intrinsic_params = [
    {
        'id': '1',
        'center': [956.6417236328125,526.0769653320312],
        'focal_length': [1579.549560546875,1580.794921875],
        'radial_distortion': [-0.14646318554878235,0.13881948590278625,-0.07125651836395264],     ## k1 k2 k3
        'tangential_distortion': [-0.0009389497572556138,-2.3357843019766733e-05], ## p1 p2
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': 0, # Only used for visualization
    },
    {
        'id': '2',
        'center': [959.6974487304688,545.5534057617188],
        'focal_length': [1590.145263671875,1589.552734375],
        'radial_distortion': [-0.1452776938676834,0.13771952688694,-0.06734155863523483], ## k1 k2 k3
        'tangential_distortion': [-0.0008101108833216131,0.00038700600271113217], ## p1 p2
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': 0, # Only used for visualization
    },
    {
        'id': '3',
        'center': [945.036376953125,544.3353271484375],
        'focal_length': [1588.7041015625,1587.5155029296875],
        'radial_distortion': [-0.13589085638523102,0.10075731575489044,-0.015615474432706833],     ## k1 k2 k3
        'tangential_distortion': [-0.0011635348200798035,-0.0009321411489509046], ## p1 p2
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': 0, # Only used for visualization
    },
    {
        'id': '4',
        'center': [945.07275390625,553.9684448242188],
        'focal_length': [1590.0548095703125,1590.5556640625],
        'radial_distortion': [-0.13788962364196777,0.10029269009828568,-0.014170068316161633],     ## k1 k2 k3
        'tangential_distortion': [-0.0008639143197797239,-0.0013395894784480333], ## p1 p2
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': 0, # Only used for visualization
    },
    {
        'id': '5',
        'center': [958.4230346679688,547.2905883789062],
        'focal_length': [1590.8802490234375,1590.1357421875],
        'radial_distortion': [-0.14894536137580872,0.1595952957868576,-0.10301978141069412],     ## k1 k2 k3
        'tangential_distortion': [-0.0006442437297664583,7.107679994078353e-05,], ## p1 p2
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': 0, # Only used for visualization
    },
    {
        'id': '6',
        'center': [957.1611938476562,560.1165771484375],
        'focal_length': [1592.3768310546875,1591.3646240234375],
        'radial_distortion': [-0.1426018476486206,0.12450122088193893,-0.032923612743616104],     ## k1 k2 k3
        'tangential_distortion': [-0.000379784032702446,-0.0004391535185277462], ## p1 p2
        'res_w': 1920,
        'res_h': 1080,
        'azimuth': 0, # Only used for visualization
    }
]

h36m_cameras_extrinsic_params = {
    'S1': [
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    ],
    'S2': [
        {
            'orientation': [0.7214346772808389, 0.5594938250505211, 0.2669336309336537, 0.30861805330149544],
            'translation': [-2576.6845703125, 2967.963134765625, -1642.8984375],
        },
        {
            'orientation': [-0.41221846188522615, -0.3176995114936784, -0.5228560724698434, -0.675103316210983],
            'translation': [-2950.165771484375, -1363.7650146484375, -1500.605224609375],
        },
        {
            'orientation': [0.667356265136803, 0.5323901896765427, -0.3143188082645252, -0.4152107754984217],
            'translation': [4275.3818359375, 2391.6259765625, -1626.4405517578125],
        },
        {
            'orientation': [0.792664258130462, 0.6093196078287898, -0.002031477453301868, -0.02022034859271058],
            'translation': [575.7633056640625, 4155.89599609375, -1636.170654296875],
        },
        {
            'orientation': [0.3228321571449658, 0.28061021797282065, -0.5607030876914877, -0.7089776804213276],
            'translation': [4303.7021484375, -2368.0400390625, -1624.0472412109375],
        },
        {
            'orientation': [-0.024755272009995985, -0.02000983248580507, -0.6137013188067142, -0.7888963648075079],
            'translation': [221.23043823242188, -4081.730712890625, -1636.85107421875],
        }
    ],
    'S3': [
        {},
        {},
        {},
        {},
    ],
    'S4': [
        {},
        {},
        {},
        {},
    ],
    'S5': [
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    'S6': [
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    'S7': [
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    'S8': [
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    'S9': [
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    'S11': [
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}


class Human36mDataset(MocapDataset):
    def __init__(self, path3d, path2d, remove_static_joints=True, chunk_length=1, augment=False, pad=121, causal_shift=0, is_train=False):
        super().__init__(fps=50, skeleton=copy.deepcopy(h36m_skeleton))
        self.chunk_length = chunk_length
        self.augment = augment
        self.pad = pad
        self.causal_shift = causal_shift
        self.is_train = is_train
        self.train_subjects = ["S1", "S2", "S5", "S6", "S7", "S8"]
        self.test_subjects = ["S9", "S11"]

        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for sub,cameras in self._cameras.items():
            for i, cam in enumerate(cameras):
                if sub == "S2":
                    cam.update(h36m_goProCameras_intrinsic_params[i])
                else:
                    cam.update(h36m_cameras_intrinsic_params[i])
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

        self.process3DPose()
        self.normalize2DPose()
        # self._pairs = self._data_2d["S1"]["Directions"][0]
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
                    pos_3d[:, :] = np.mean(pos_3d[:, 11:13],axis = 1,keepdims=True)  # Remove global offset, but keep trajectory in first position
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

        subject, action, camIndex = subject_action_camIndex.split("_")
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
