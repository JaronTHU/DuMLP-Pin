#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
======================
@author: Jaron
@time: 2021/10/6:16:27
@email: fjjth98@163.com
@description: 
======================
"""

import os
import torch
import logging
import argparse
import torch.nn as nn

from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils import SoftCrossEntropyLoss


if __name__ == '__main__':

    # parse the arguments
    parser = argparse.ArgumentParser('DuMLP-Pin training and evaluation script', add_help=False)
    parser.add_argument('--mode', required=True, choices=['train', 'eval'])
    parser.add_argument('--task', required=True, choices=['ModelNet40', 'ShapeNetPart'])
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--learning-rate-milestones', type=str, help='use , to separate numbers (no space)')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma rate for MultiStepLR')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--soft-cross-entropy', type=float, default=0., help='0. means cross entropy')
    parser.add_argument('--max-epoch', type=int, default=-1, help='-1 means no stop')
    parser.add_argument('--model-cls', type=str, required=True)
    parser.add_argument('--model-save-dir', type=str, help='')
    parser.add_argument('--model-load-path', type=str)
    parser.add_argument('--data-cls', type=str, required=True)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--gpu', type=int, default=-1, help='-1 = cpu')
    parser.add_argument('--random-seed', type=int, default=42)
    args, unparsed = parser.parse_known_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    exec('from MyModel import {} as MyModel'.format(args.model_cls))
    exec('from MyDataset import {} as MyDataset'.format(args.data_cls))
    torch.manual_seed(args.random_seed)
    if args.gpu == -1:
        use_cuda = False
        device = torch.device('cpu')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        use_cuda = True
        device = torch.device('cuda')
    model = MyModel().to(device)
    if args.model_load_path:
        model.load_state_dict(torch.load(args.model_load_path, map_location=device)['model'])
    if args.model_save_dir:
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        writer = SummaryWriter(log_dir=args.model_save_dir)
        fh = logging.FileHandler(os.path.join(args.model_save_dir, 'output.log'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if args.mode == 'train':
        loss_fn = nn.CrossEntropyLoss().to(device) if args.soft_cross_entropy == 0. else SoftCrossEntropyLoss().to(
            device)
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[int(a) for a in args.learning_rate_milestones.split(',')], gamma=args.gamma)

    if args.task == 'ModelNet40':
        from utils import get_tnf, tnf2met, write_met
        eval_set = MyDataset(root=args.data_dir, train=False, augment=False, cuda=use_cuda)
        eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)
        if args.model_save_dir:
            writer.add_graph(model, torch.randn(1, 1024, 6, device=device))
        num_cats, name_cats = len(MyDataset.classes), MyDataset.classes
        if args.mode == 'train':
            epoch, best_met, best_epoch, record_dict = 0, -float('inf'), 0, {}
            train_set = MyDataset(root=args.data_dir, train=True, augment=True, cuda=use_cuda)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            while epoch != args.max_epoch:
                model.train()
                train_tnf = torch.zeros((num_cats, 2), dtype=torch.long, device=device)
                train_loss = 0.
                for inputs, labels in tqdm(train_loader):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * labels.shape[0]
                    get_tnf(outputs, labels, train_tnf)
                model.eval()
                with torch.no_grad():
                    eval_tnf = torch.zeros((num_cats, 2), dtype=torch.long, device=device)
                    eval_loss = 0.
                    for inputs, labels in tqdm(eval_loader):
                        outputs = model(inputs)
                        eval_loss += loss_fn(outputs, labels).item() * labels.shape[0]
                        get_tnf(outputs, labels, eval_tnf)
                epoch += 1
                scheduler.step()
                my_info = 'Epoch: {:<3d}'.format(epoch)
                for mode in ['eval', 'train']:
                    exec('record_dict["{}_loss"] = {}_loss / len({}_set)'.format(mode, mode, mode))
                    exec('record_dict["{}_oa"], record_dict["{}_macc"], record_dict["{}_acc"] = tnf2met({}_tnf)'.format(mode, mode, mode, mode))
                    my_info += '; {} OA: {:.2%}; {} mAcc: {:.2%}'.format(mode, record_dict['{}_oa'.format(mode)], mode, record_dict['{}_macc'.format(mode)])
                logger.info(my_info)
                if args.model_save_dir:
                    write_met(writer, epoch, record_dict, MyDataset.classes)
                if record_dict['eval_oa'] > best_met:
                    best_met, best_epoch = record_dict['eval_oa'], epoch
                    logger.info('New best model is achieved at epoch {} with eval OA as {:.2%}'.format(best_epoch, best_met))
                    if args.model_save_dir:
                        torch.save({'epoch': epoch, 'metric': best_met, 'model': model.state_dict()}, os.path.join(args.model_save_dir, 'best_model.pth'))
            logger.info('Best model is achieved at epoch {} with eval OA as {:.2%}'.format(best_epoch, best_met))
            if args.model_save_dir:
                torch.save({'epoch': epoch, 'metric': record_dict['eval_oa'], 'model': model.state_dict()}, os.path.join(args.model_save_dir, 'final_model.pth'))
                writer.close()
        else:
            model.eval()
            with torch.no_grad():
                eval_tnf = torch.zeros((num_cats, 2), dtype=torch.long, device=device)
                for inputs, labels in tqdm(eval_loader):
                    get_tnf(model(inputs), labels, eval_tnf)
                oa, macc, _ = tnf2met(eval_tnf)
                logger.info('eval OA: {:.2%}; eval mAcc: {:.2%}'.format(oa, macc))
    elif args.task == 'ShapeNetPart':
        from utils import get_iou, tnf2met, write_met
        eval_set = MyDataset(root=args.data_dir, train=False, augment=False, cuda=use_cuda)
        eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)
        num_cats, name_cats = len(MyDataset.classes), MyDataset.classes
        if args.model_save_dir:
            writer.add_graph(model, [torch.randn(1, 2048, 3, device=device), torch.randint(0, num_cats, (1,), device=device)])
        if args.mode == 'train':
            epoch, best_met, best_epoch, record_dict = 0, -float('inf'), 0, {}
            train_set = MyDataset(root=args.data_dir, train=True, augment=True, cuda=use_cuda)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            while epoch != args.max_epoch:
                model.train()
                train_iou = torch.zeros((num_cats, 2), dtype=torch.float, device=device)
                train_loss = 0.
                for inputs, labels, pids in tqdm(train_loader):
                    outputs = model(inputs, labels)
                    loss = loss_fn(outputs.view(-1, 50), pids.view(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * labels.shape[0]
                    get_iou(outputs, labels, pids, train_iou)
                model.eval()
                with torch.no_grad():
                    eval_iou = torch.zeros((num_cats, 2), dtype=torch.float, device=device)
                    eval_loss = 0.
                    for inputs, labels, pids in tqdm(eval_loader):
                        outputs = model(inputs, labels)
                        eval_loss += loss_fn(outputs.view(-1, 50), pids.view(-1)).item() * labels.shape[0]
                        get_iou(outputs, labels, pids, eval_iou)
                epoch += 1
                scheduler.step()
                my_info = 'Epoch: {:<3d}'.format(epoch)
                for mode in ['eval', 'train']:
                    exec('record_dict["{}_loss"] = {}_loss / len({}_set)'.format(mode, mode, mode))
                    exec('record_dict["{}_piou"], record_dict["{}_ciou"], record_dict["{}_iou"] = tnf2met({}_iou)'.format(mode, mode, mode, mode))
                    my_info += '; {} pIoU: {:.2%}; {} cIoU: {:.2%}'.format(mode, record_dict['{}_piou'.format(mode)], mode, record_dict['{}_ciou'.format(mode)])
                logger.info(my_info)
                if args.model_save_dir:
                    write_met(writer, epoch, record_dict, MyDataset.classes)
                if record_dict['eval_piou'] > best_met:
                    best_met, best_epoch = record_dict['eval_piou'], epoch
                    logger.info('New best model is achieved at epoch {} with eval pIoU as {:.2%}'.format(best_epoch, best_met))
                    if args.model_save_dir:
                        torch.save({'epoch': epoch, 'metric': best_met, 'model': model.state_dict()}, os.path.join(args.model_save_dir, 'best_model.pth'))
            logger.info('Best model is achieved at epoch {} with eval pIoU as {:.2%}'.format(best_epoch, best_met))
            if args.model_save_dir:
                torch.save({'epoch': epoch, 'metric': record_dict['eval_piou'], 'model': model.state_dict()}, os.path.join(args.model_save_dir, 'final_model.pth'))
                writer.close()
        else:
            model.eval()
            with torch.no_grad():
                eval_iou = torch.zeros((num_cats, 2), dtype=torch.float, device=device)
                for inputs, labels, pids in tqdm(eval_loader):
                    get_iou(model(inputs, labels), labels, pids, eval_iou)
                piou, ciou, _ = tnf2met(eval_iou)
                logger.info('eval pIoU: {:.2%}; eval cIoU: {:.2%}'.format(piou, ciou))
