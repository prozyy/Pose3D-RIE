# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from common.skeleton import Skeleton
from torch.utils.data import Dataset

class MocapDataset(Dataset):
    def __init__(self, fps, skeleton):
        self._skeleton = skeleton
        self._fps = fps
        self._data_3d = None # Must be filled by subclass
        self._data_2d = None # Must be filled by subclass
        self._cameras = None # Must be filled by subclass
        self._pairs = []
        self.kept_joints = None
    
    def remove_joints(self, joints_to_remove):
        self.kept_joints = self._skeleton.remove_joints(joints_to_remove)
                
    def __getitem__(self, index):
        return self.getDataFromPair(index)

    def __len__(self):
        return len(self._pairs)
        
    def subjects(self):
        return self._data_3d.keys()
    
    def fps(self):
        return self._fps
    
    def skeleton(self):
        return self._skeleton
        
    def cameras(self):
        return self._cameras
    
    def supports_semi_supervised(self):
        # This method can be overridden
        return False

    def getDataFromPair(self,index):
        return None
