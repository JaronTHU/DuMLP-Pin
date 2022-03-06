#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
@author: Jaron
@time: 2021/4/27:18:33
@email: fjjth98@163.com
@description: Layers
======================
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, Optional


def name2act(name: Optional[str] = None):
    activation = None
    if name == 'relu':
        activation = nn.ReLU(inplace=True)
    elif name == 'elu':
        activation = nn.ELU(inplace=True)
    elif name == 'none' or name is None:
        activation = nn.Identity()
    elif name == 'softmax':
        activation = nn.Softmax(dim=1)
    elif name == 'squashing':
        activation = MySquash(dim=1)
    return activation


class MySquash(nn.Module):
    r"""Applies squashing to the incoming data: y=||x|| / (1 + ||x||^2) * x
    From: Dynamic Routing Between Capsules
    https://proceedings.neurips.cc/paper/2017/file/2cad8fa47bbef282badbb8de5374b894-Paper.pdf
    """

    def __init__(self, dim: int) -> None:
        super(MySquash, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x_norm = torch.linalg.norm(x, dim=self.dim, keepdim=True)
        return x_norm * (1. + x_norm * x_norm) * x

    def extra_repr(self) -> str:
        return 'dim={}'.format(self.dim)


class MyLinear(nn.Module):
    r"""Applies linear + batchnorm / dropout + activation to the incoming data
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: Optional[str] = None, dropout: Optional[float] = None) -> None:
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.li = nn.Linear(in_features, out_features, bias)
        self.db = nn.BatchNorm1d(out_features) if dropout is None else nn.Dropout(dropout, inplace=True)
        self.ac = name2act(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.li(x)
        x_shape = x.shape
        return self.ac(self.db(x.view(-1, x_shape[-1])).view(x_shape))

    def extra_repr(self) -> str:
        db = 'BatchNorm' if str(self.db)[0] == 'B' else str(self.db).split(',')[0] + ')'
        return 'in_features={}, out_features={}, bias={}, activation={}, {}'.format(
            self.in_features, self.out_features, self.li.bias is not None, str(self.ac), db
        )


class MyBilinear(nn.Module):
    r"""Applies my bilinear + batchnorm / dropout + activation to the incoming data
    HUGE difference from torch.nn.Bilinear !!!
    torch.nn.Bilinear: x_1: (B, *, d_1), x_2: (B, *, d_2), y=x_1^TAx_2+b
    MyBilinear: x_1: (B, N, d_1), x_2: (B, d_2), y=(x_1^TAx_2/0)+x_1^TA_1+x_2^TA_2+b
    """

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True,
                 activation: Optional[str] = None, dropout: Optional[float] = None,
                 use_cross_term: bool = False) -> None:
        super(MyBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.li1 = nn.Linear(in1_features, out_features, bias)
        self.li2 = nn.Linear(in2_features, out_features, bias=False)
        self.db = nn.BatchNorm1d(out_features) if dropout is None else nn.Dropout(dropout, inplace=True)
        self.ac = name2act(activation)
        if use_cross_term:
            self.bli = nn.Parameter(torch.Tensor(in2_features, in1_features * out_features))
            bound = 1. / math.sqrt(in1_features)
            torch.nn.init.uniform_(self.bli, -bound, bound)
        else:
            self.bli = None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = self.li1(x) + self.li2(y).unsqueeze(1)
        if self.bli is not None:
            res = res + torch.matmul(x, torch.matmul(y, self.bli).view(-1, self.in1_features, self.out_features))
        res_shape = res.shape
        return self.ac(self.db(res.view(-1, res_shape[-1])).view(res_shape))

    def extra_repr(self) -> str:
        db = 'batchnorm' if str(self.db)[0] == 'B' else str(self.db).split(',')[0] + ')'
        return 'in1_features={}, in2_features={}, out_features={}, bias={}, use_cross_term={}, activation={}, {}'.format(
            self.in1_features, self.in2_features, self.out_features,
            self.li1.bias is not None, self.bli is not None, str(self.ac), db
        )


class MyMLP(nn.Module):
    r"""Stacks of MyLinear
    For last layer of classification, no norm are used, otherwise use.
    """

    def __init__(self, unit_list: Tuple[int, ...], activation: Optional[str] = 'relu', dropout: Optional[float] = None,
                 use_norm_last_layer: bool = True) -> None:
        super(MyMLP, self).__init__()
        self.unit_list = unit_list
        self.activation = activation
        self.db = 'BatchNorm' if dropout is None else 'Dropout(p={})'.format(dropout)
        assert len(unit_list) > 1, 'MLP must have at least 1 layer!'
        self.layers = nn.ModuleList()
        for i in range(len(unit_list) - 2):
            self.layers.append(MyLinear(unit_list[i], unit_list[i + 1], activation=activation, dropout=dropout))
        if use_norm_last_layer:
            self.layers.append(MyLinear(unit_list[-2], unit_list[-1], activation=None, dropout=dropout))
        else:
            self.layers.append(nn.Linear(unit_list[-2], unit_list[-1]))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def extra_repr(self) -> str:
        return 'unit_list={}, activation={}, {}'.format(self.unit_list, self.activation, self.db)


class AggregationBlock(nn.Module):
    r"""Basic 2-way Set Absorb Block
    """

    def __init__(self, in_features: int, unit_lists: Tuple[Tuple[int, ...], Tuple[int, ...]],
                 activation: Union[str, Tuple[str, str]], reg: float = 1.) -> None:
        super(AggregationBlock, self).__init__()
        self.in_features = in_features
        self.out_features = unit_lists[0][-1] * unit_lists[1][-1]
        self.unit_lists = tuple([(in_features,) + ul for ul in unit_lists])
        self.mlps = nn.ModuleList([MyMLP(ul, activation='relu') for ul in self.unit_lists])
        if isinstance(activation, str):
            activation = (activation, activation)
        self.activations = nn.ModuleList([name2act(act) for act in activation])
        self.reg = reg

    def forward(self, x: Tensor) -> Tensor:
        y = [activation(mlp(x)) for activation, mlp in zip(self.activations, self.mlps)]
        return self.reg * torch.matmul(y[0].transpose(1, 2), y[1]).flatten(1)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}\nmlp1: unit_list={}, activation={}\nmlp2: unit_list={}, activation={}'.format(
            self.in_features, self.out_features, self.unit_lists[0], str(self.activations[0]),
            self.unit_lists[1], str(self.activations[1])
        )


class AggregationBlockN(nn.Module):
    r"""N-way Set Absorb Block
    reg^n
    """

    def __init__(self, in_features: int, unit_lists: Tuple[Tuple[int, ...], ...],
                 activation: Union[str, Tuple[str, ...]], reg: float = 1.) -> None:
        super(AggregationBlockN, self).__init__()
        self.in_features = in_features
        self.unit_lists = tuple([(in_features,) + ul for ul in unit_lists])
        self.mlps = nn.ModuleList([MyMLP(ul, activation='relu') for ul in self.unit_lists])
        if isinstance(activation, str):
            activation = (activation,) * len(self.unit_lists)
        self.activations = nn.ModuleList([name2act(act) for act in activation])
        self.reg = reg
        self.out_features = unit_lists[0][-1]
        for i in range(1, len(unit_lists)):
            self.out_features *= unit_lists[i][-1]

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        res = self.reg * self.activations[0](self.mlps[0](x)).view(B * N, -1, 1)
        for i in range(1, len(self.unit_lists)):
            y = self.reg * self.activations[i](self.mlps[i](x)).view(B * N, 1, -1)
            res = torch.matmul(res, y).view(B * N, -1, 1)
        return res.view(B, N, -1).sum(dim=1)

    def extra_repr(self) -> str:
        ss = 'in_features={}, out_features={}'.format(self.in_features, self.out_features)
        for i in range(len(self.unit_lists)):
            ss += '\nmlp{}: unit_list={}, activation={}'.format(i + 1, self.unit_lists[i], str(self.activations[i]))
        return ss


class BroadcastBlock(nn.Module):

    def __init__(self, unit_list_e: Tuple[int, ...], unit_list_s: Tuple[int, ...], unit_list_a: Tuple[int, ...],
                 use_cross_term: bool = False) -> None:
        super(BroadcastBlock, self).__init__()
        self.unit_list_e = unit_list_e
        self.unit_list_s = unit_list_s
        self.unit_list_a = unit_list_a
        self.use_cross_term = use_cross_term
        self.mlp_e = MyMLP(unit_list_e) if len(unit_list_e) > 1 else nn.Identity()
        self.mlp_s = MyMLP(unit_list_s) if len(unit_list_s) > 1 else nn.Identity()
        self.comb = MyBilinear(unit_list_e[-1], unit_list_s[-1], unit_list_a[0],
                               activation='relu', use_cross_term=use_cross_term)
        self.mlp_a = MyMLP(unit_list_a) if len(unit_list_a) > 1 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if len(self.unit_list_e) > 1:
            x = self.relu(self.mlp_e(x))
        if len(self.unit_list_s) > 1:
            y = self.relu(self.mlp_s(y))
        x = self.comb(x, y)
        if len(self.unit_list_a) > 1:
            x = self.relu(self.mlp_a(x))
        return x

    def extra_repr(self) -> str:
        return 'element mlp: unit_list={}, set mlp: unit_list={}, all mlp: unit_list={}, use_cross_term={}'.format(self.unit_list_e, self.unit_list_s, self.unit_list_a, self.use_cross_term)
