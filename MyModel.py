#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
@author: Jaron
@time: 2021/10/8:11:46
@email: fjjth98@163.com
@description: 
======================
"""

import torch
import torch.nn as nn
from MyLayer import MyMLP, AggregationBlock, BroadcastBlock


class ModelNet40(nn.Module):

    def __init__(self):
        super(ModelNet40, self).__init__()
        self.sab = AggregationBlock(6, ((32, 128, 32), (32, 128, 32)), activation=('softmax', 'none'))
        self.mlp = MyMLP((1024, 256, 40), use_norm_last_layer=False)

    def forward(self, x):
        x = self.sab(x)
        x = self.mlp(x)
        return x


class ShapeNetPartS(nn.Module):

    def __init__(self):
        super(ShapeNetPartS, self).__init__()
        self.sau_i = BroadcastBlock((3, 32, 128), (16, 128, 128), (128, 128))
        self.sabs, self.saus = nn.ModuleList(), nn.ModuleList()
        for _ in range(0):
            self.sabs.append(AggregationBlock(128, ((128, 16), (128, 16)), activation=('softmax', 'none')))
            self.saus.append(BroadcastBlock((128, 128), (256, 128), (128, 128)))
        self.sab_e = AggregationBlock(128, ((128, 16), (128, 16)), activation=('softmax', 'none'))
        self.sau_e = BroadcastBlock((128, 128), (256, 128), (1024, 256))
        self.li = nn.Linear(256, 50)

    def forward(self, x, y):
        x = self.sau_i(x, torch.eye(16, dtype=torch.float, device=y.device)[y])
        for sab, sau in zip(self.sabs, self.saus):
            y = sab(x)
            x = sau(x, y)
        y = self.sab_e(x)
        x = self.sau_e(x, y)
        x = self.li(x)
        return x
    

class ShapeNetPartL(nn.Module):

    def __init__(self):
        super(ShapeNetPartL, self).__init__()
        self.sau_i = BroadcastBlock((3, 32, 128), (16, 128, 128), (128, 128))
        self.sabs, self.saus = nn.ModuleList(), nn.ModuleList()
        for _ in range(2):
            self.sabs.append(AggregationBlock(128, ((128, 16), (128, 16)), activation=('softmax', 'none')))
            self.saus.append(BroadcastBlock((128, 128), (256, 128), (128, 128)))
        self.sab_e = AggregationBlock(128, ((128, 16), (128, 16)), activation=('softmax', 'none'))
        self.sau_e = BroadcastBlock((128, 128), (256, 128), (1024, 256))
        self.li = nn.Linear(256, 50)

    def forward(self, x, y):
        x = self.sau_i(x, torch.eye(16, dtype=torch.float, device=y.device)[y])
        for sab, sau in zip(self.sabs, self.saus):
            y = sab(x)
            x = sau(x, y)
        y = self.sab_e(x)
        x = self.sau_e(x, y)
        x = self.li(x)
        return x
