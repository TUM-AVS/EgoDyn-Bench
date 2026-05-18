# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu, CMU
# All rights reserved.
#
# Vendored from https://github.com/castacks/tartanvo

import torch
import torch.nn as nn
from .PWCNet import PWCDCNet as FlowNet
from .VOFlowNet import VOFlowRes as FlowPoseNet

class VONet(nn.Module):
    def __init__(self):
        super(VONet, self).__init__()

        self.flowNet     = FlowNet()
        self.flowPoseNet = FlowPoseNet()

    def forward(self, x):
        flow = self.flowNet(x[0:2])
        flow_input = torch.cat( ( flow, x[2] ), dim=1 )
        pose = self.flowPoseNet( flow_input )

        return flow, pose
