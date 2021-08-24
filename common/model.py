# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from common.models.rieModel import RIEModel
from common.models.rieModel24 import RIEModel as RIEModel24


def get_model(dataSetType="h36m"):
    if dataSetType == "h36m":
        return RIEModel
    elif dataSetType == "h36m_24":
        return RIEModel24
    else:
        raise ("unknow dataSetType: " + str(dataSetType))
