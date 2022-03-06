#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
@author: Jaron
@time: 2021/10/8:11:49
@email: fjjth98@163.com
@description: 
======================
"""

import os
import torch

from typing import Tuple, Any
from torch.utils.data import Dataset


class MyModelNet40(Dataset):

    classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    def __init__(
            self,
            root: str,
            train: bool = True,
            random_sample: bool = False,
            sample_num: int = 1024,
            use_norm: bool = True,
            augment: bool = True,
            cuda: bool = True,
            seed: int = 2
    ) -> None:
        super(MyModelNet40, self).__init__()
        self.train = train
        self.random_sample = random_sample
        self.augment = augment

        if not os.path.exists(root):
            os.makedirs(root)
            os.system('wget -P {} --no-check-certificate https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip'.format(root))
            os.system('unzip -q {}/modelnet40_normal_resampled.zip -d {}'.format(root, root))
            cat_dict = dict(zip(MyModelNet40.classes, range(len(MyModelNet40.classes))))
            for split in ('test', 'train'):
                data, label = [], []
                with open('{}/modelnet40_normal_resampled/modelnet40_{}.txt'.format(root, split), 'r') as f:
                    file_list = [i[:i.rfind('_')] + '/' + i[:-1] + '.txt' for i in f.readlines()]
                for file in file_list:
                    with open('{}/modelnet40_normal_resampled/{}'.format(root, file), 'r') as f:
                        data.append(torch.tensor([list(map(float, i[:-1].split(','))) for i in f.readlines()]))
                        label.append(cat_dict[file.split('/')[0]])
                data = torch.stack(data, dim=0)
                label = torch.tensor(label, dtype=torch.uint8)
                torch.save((data, label), '{}/{}.pt'.format(root, 'test' if split == 'test' else 'training'))
            os.system('rm -rf {}/modelnet40_normal_resampled'.format(root))
            os.system('rm {}/modelnet40_normal_resampled.zip'.format(root))

        data_file = os.path.join(root, 'training.pt') if self.train else os.path.join(root, 'test.pt')
        self.data, self.targets = torch.load(data_file)
        self.targets = self.targets.to(torch.long)

        if random_sample:
            self.sample_num = sample_num
        else:
            self.data = self.data[:, :sample_num]
            self.data[..., :3] -= self.data[..., :3].mean(dim=1, keepdims=True)
            self.data[..., :3] /= torch.max(torch.linalg.norm(self.data[..., :3], dim=2, keepdim=True), dim=1, keepdim=True)[0]

        if not use_norm:
            self.data = self.data[..., :3]

        if cuda:
            self.data, self.targets = self.data.cuda(), self.targets.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pc, target = self.data[index].clone(), self.targets[index]
        if self.random_sample:
            pc = pc[torch.randperm(self.data.shape[1], device=self.device, generator=self.generator)[:self.sample_num]]
            pc[:, :3] -= pc[:, :3].mean(dim=0)
            pc[:, :3] /= torch.max(torch.linalg.norm(pc[:, :3], dim=1))
        if self.augment:
            self.drop(pc)
            self.scale(pc)
            self.shift(pc)
            self.jitter(pc)
            self.rotate(pc)
        return pc, target

    def drop(self, pc: torch.Tensor, low: float = 0., high: float = 0.875) -> None:
        non_drop_idx = []
        while len(non_drop_idx) == 0:
            dropout_ratio = torch.empty(1, device=self.device).uniform_(low, high, generator=self.generator)
            drop_bool = torch.rand(pc.shape[0], device=self.device, generator=self.generator) <= dropout_ratio
            drop_idx, non_drop_idx = torch.where(drop_bool)[0], torch.where(~drop_bool)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[non_drop_idx[torch.randint(len(non_drop_idx), (len(drop_idx),), device=self.device, generator=self.generator)]]

    def scale(self, pc: torch.Tensor, low: float = 0.8, high: float = 1.25) -> None:
        pc[:, :3] *= torch.empty(1, device=self.device).uniform_(low, high, generator=self.generator)

    def shift(self, pc: torch.Tensor, scale: float = 0.1) -> None:
        pc[:, :3] += torch.empty(3, device=self.device).uniform_(-scale, scale, generator=self.generator)

    def jitter(self, pc: torch.Tensor, sigma: float = 0.01, clip: float = 0.05) -> None:
        pc += torch.clip(torch.empty_like(pc).normal_(std=sigma, generator=self.generator), -clip, clip)
        if pc.shape[1] == 6:
            pc[:, 3:] /= torch.linalg.norm(pc[:, 3:], dim=1, keepdims=True)

    def rotate(self, pc: torch.Tensor, sigma: float = 0.06, clip: float = 0.18) -> None:
        angles = torch.clip(torch.empty(3, device=self.device).normal_(std=sigma, generator=self.generator), -clip, clip)
        cosa, sina = torch.cos(angles), torch.sin(angles)
        Rx = torch.tensor([[1., 0., 0.], [0., cosa[0], -sina[0]], [0., sina[0], cosa[0]]], device=self.device)
        Ry = torch.tensor([[cosa[1], 0., -sina[1]], [0., 1., 0.], [sina[1], 0., cosa[1]]], device=self.device)
        Rz = torch.tensor([[cosa[2], -sina[2], 0.], [sina[2], cosa[2], 0.], [0., 0., 1.]], device=self.device)
        R = torch.matmul(Rz, torch.matmul(Ry, Rx))
        pc[:, :3] = torch.matmul(pc[:, :3], R)
        if pc.shape[1] == 6:
            pc[:, 3:] = torch.matmul(pc[:, 3:], R)

    def __len__(self) -> int:
        return len(self.data)


class MyShapeNetPart(Dataset):

    classes = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']

    parts = [[0, 1, 2, 3], [4, 5], [6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23], [24, 25, 26, 27], [28, 29], [30, 31, 32, 33, 34, 35], [36, 37], [38, 39, 40], [41, 42, 43], [44, 45, 46], [47, 48, 49]]

    def __init__(
            self,
            root: str,
            train: bool = True,
            augment: bool = True,
            cuda: bool = True,
            seed: int = 30
    ) -> None:
        super(MyShapeNetPart, self).__init__()
        self.train = train
        self.augment = augment

        if not os.path.exists(root):
            import h5py
            os.makedirs(root)
            os.system('wget -P {} --no-check-certificate https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'.format(root))
            os.system('unzip -q {}/shapenet_part_seg_hdf5_data.zip -d {}'.format(root, root))
            result = {}
            for split, n in zip(('val', 'test', 'train'), (1, 2, 6)):
                data, label, pid = [], [], []
                for i in range(n):
                    with h5py.File('{}/hdf5_data/ply_data_{}{}.h5'.format(root, split, i), 'r') as f:
                        data.append(torch.tensor(f['data'][:]))
                        label.append(torch.tensor(f['label'][:, 0]))
                        pid.append(torch.tensor(f['pid'][:]))
                data = torch.cat(data, dim=0)
                label = torch.cat(label, dim=0)
                pid = torch.cat(pid, dim=0)
                result[split] = (data, label, pid)
            torch.save(result['test'], '{}/test.pt'.format(root))
            torch.save((
                torch.cat((result['train'][0], result['val'][0]), dim=0),
                torch.cat((result['train'][1], result['val'][1]), dim=0),
                torch.cat((result['train'][2], result['val'][2]), dim=0)
            ), '{}/training.pt'.format(root))
            os.system('rm -rf {}/hdf5_data'.format(root))
            os.system('rm -rf {}/shapenet_part_seg_hdf5_data.zip'.format(root))

        data_file = os.path.join(root, 'training.pt') if self.train else os.path.join(root, 'test.pt')
        self.data, self.labels, self.pids = torch.load(data_file)
        self.labels, self.pids = self.labels.to(torch.long), self.pids.to(torch.long)

        if cuda:
            self.data, self.labels, self.pids = self.data.cuda(), self.labels.cuda(), self.pids.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, label, pid = self.data[index].clone(), self.labels[index], self.pids[index]
        if self.augment:
            self.shift(data)
            self.scale(data)
        return data, label, pid

    def __len__(self) -> int:
        return len(self.data)

    def scale(self, pc: torch.Tensor, low: float = 0.67, high: float = 1.5) -> None:
        pc *= torch.empty(1, device=self.device).uniform_(low, high, generator=self.generator)

    def shift(self, pc: torch.Tensor, scale: float = 0.2) -> None:
        pc += torch.empty(3, device=self.device).uniform_(-scale, scale, generator=self.generator)
