#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
@author: Jaron
@time: 2021/10/7:11:57
@email: fjjth98@163.com
@description: 
======================
"""

import torch
import torch.nn as nn

from MyDataset import MyShapeNetPart


class SoftCrossEntropyLoss(nn.Module):
    r"""Soft cross entropy loss
    input: (N, C), target: (N, ) each value 0<=...<=C - 1
    """

    def __init__(self, ep: float = 0.2, reduction: str = 'mean'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.ep = ep
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N, C = input.shape
        target_one_hot = torch.full_like(input, self.ep / (C - 1))
        target_one_hot.scatter_(1, target.unsqueeze(1), 1 - self.ep)
        log_prob = self.log_softmax(input)
        loss = -(target_one_hot * log_prob).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def extra_repr(self) -> str:
        return 'ep={}, weight={}, reduction={}'.format(
            self.ep, self.weight is not None, self.reduction
        )


def get_tnf(outputs, labels, tnf):
    """
    tnf (true and false) count
    :param outputs:
    :param labels:
    :param tnf:
    :return:
    """
    predictions = torch.argmax(outputs, dim=-1)
    for i in range(tnf.shape[0]):
        mid1, mid2 = labels == i, predictions == i
        tnf[i, 0] += (mid1 & mid2).sum()
        tnf[i, 1] += mid1.sum()


def tnf2met(tnf):
    """
    tnf to metric for ModelNet40
    :param tnf:
    :return:
    """
    oa = tnf[:, 0].sum() / tnf[:, 1].sum()
    acc = tnf[:, 0] / tnf[:, 1]
    macc = acc.mean()
    return oa, macc, acc


def write_met(writer, epoch, record_dict, name_cats=None):
    """
    :param writer:
    :param epoch:
    :param record_dict:
    :param name_cats: category acc or iou
    :return:
    """
    name_mets = [k[6:] for k in record_dict.keys() if k[:5] == 'train']

    if name_cats is not None:
        if 'acc' in name_mets:
            cat_met = 'acc'
        elif 'iou' in name_mets:
            cat_met = 'iou'
        name_mets.remove(cat_met)
        for mode in ['train', 'eval']:
            for i, name in enumerate(name_cats):
                writer.add_scalar('{}/{}'.format(name, mode), record_dict['{}_{}'.format(mode, cat_met)][i], epoch)
    writer.flush()
    for mode in ['train', 'eval']:
        for met in name_mets:
            writer.add_scalar('{}/{}'.format(met, mode), record_dict['{}_{}'.format(mode, met)], epoch)


def get_iou(outputs, labels, pids, iou):
    """
    IoU for ShapeNetPart
    :param outputs:
    :param labels:
    :param pids:
    :param iou:
    :return:
    """
    predictions = torch.argmax(outputs, dim=-1)
    iou_classes = torch.empty(6, device=outputs.device)
    for i in range(labels.shape[0]):
        cur_parts = MyShapeNetPart.parts[labels[i]]
        for j, k in enumerate(cur_parts):
            pk, tk = predictions[i] == k, pids[i] == k
            if pk.sum() == 0 and tk.sum() == 0:
                iou_classes[j] = 1.
            else:
                iou_classes[j] = (pk & tk).sum() / (pk | tk).sum()
        iou[labels[i], 0] += iou_classes[:len(cur_parts)].mean()
        iou[labels[i], 1] += 1.
