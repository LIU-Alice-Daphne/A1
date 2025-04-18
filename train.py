import argparse
import os
import random
from collections import OrderedDict
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import PIL
import torchvision.models.segmentation

import archs
import losses
from dataset import Dataset
from loss import loss
from metrics import iou_score
from utils import AverageMeter, str2bool
from metrics import iou_score, dice_coef, recall_s, precision_s, accuracy_s, jaccard_coef

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

"""

指定参数：
--dataset dsb2018_96 
--arch NestedUNet DeepLabV3Plus FPN SegNet DeepLabV3Plus  TransUnet BiSeNetV2 GRUUNet ResNet ResNeXtUnet18_34 RUNet2 D_UNet CUNet Double_UNet

"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    parser.add_argument('--gradient_ext', default='.png',
                        help='gradient file extension')
    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='50', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'recall': AverageMeter(),
                  'precision': AverageMeter(),
                  'accuracy': AverageMeter(),
                  'jaccard': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        # plt.imshow(np.transpose(input[0].cpu().detach().numpy(), (1, 2, 0))[:, :, 0], cmap='gray')
        # plt.show()
        # plt.imshow(np.transpose(target[0].cpu().detach().numpy(), (1, 2, 0))[:, :, 0], cmap='gray')
        # plt.show()
        # plt.imshow(np.transpose(gradient[0].cpu().detach().numpy(), (1, 2, 0))[:, :, 0], cmap='gray')
        # plt.show()
        # print(input.size())
        # print(target.size())

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)

            loss /= len(outputs)

            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)


            loss = criterion(output, target)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            recall = recall_s(output, target)
            precision = precision_s(output, target)
            accuracy = accuracy_s(output, target)
            jaccard = jaccard_coef(output, target)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['recall'].update(recall, input.size(0))
        avg_meters['precision'].update(precision, input.size(0))
        avg_meters['accuracy'].update(accuracy, input.size(0))
        avg_meters['jaccard'].update(jaccard, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('recall', avg_meters['recall'].avg),
            ('precision', avg_meters['precision'].avg),
            ('accuracy', avg_meters['accuracy'].avg),
            ('jaccard', avg_meters['jaccard'].avg),

        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('precision', avg_meters['precision'].avg),
                        ('accuracy', avg_meters['accuracy'].avg),
                        ('jaccard', avg_meters['jaccard'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'recall': AverageMeter(),
                  'precision': AverageMeter(),
                  'accuracy': AverageMeter(),
                  'jaccard': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)

                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)

                # todo transformer setr
                # seg_loss = criterion(output[-1], target)
                #
                # aux_loss_1 = criterion(output[0], target)
                # aux_loss_2 = criterion(output[1], target)
                # aux_loss_3 = criterion(output[2], target)
                # loss = seg_loss + 0.2 * aux_loss_1 + 0.3 * aux_loss_2 + 0.4 * aux_loss_3


                # todo 普通
                loss = criterion(output, target)
                # loss2 = criterion(out, gradient)
                # loss = 0.8*loss+0.2*loss2

                # regularization_loss = 0.5 * lambda_value * sum(torch.sum(param ** 2) for param in model0.parameters())
                # loss += regularization_loss
                # todo 暂时不用
                # seg_iou = iou_score(output[-1], target)
                # aux_iou_1 = iou_score(output[0], target)
                # aux_iou_2 = iou_score(output[1], target)
                # aux_iou_3 = iou_score(output[2], target)
                # iou = seg_iou + 0.2 * aux_iou_1 + 0.3 * aux_iou_2 + 0.4 * aux_iou_3
                # iou = iou_score(output[-1], target)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)
                recall = recall_s(output, target)
                precision = precision_s(output, target)
                accuracy = accuracy_s(output, target)
                jaccard = jaccard_coef(output, target)
                # iou = 0

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['recall'].update(recall, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))
            avg_meters['jaccard'].update(jaccard, input.size(0))


            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('recall', avg_meters['recall'].avg),
                ('precision', avg_meters['precision'].avg),
                ('accuracy', avg_meters['accuracy'].avg),
                ('jaccard', avg_meters['jaccard'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('precision', avg_meters['precision'].avg),
                        ('accuracy', avg_meters['accuracy'].avg),
                        ('jaccard', avg_meters['jaccard'].avg)])

# 定义模型
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.linear = nn.Linear(10, 1)
#
#     def forward(self, x):
#         return self.linear(x)
#
# model0 = MyModel()
#
# # 定义正则化强度参数
# lambda_value = 0.1

def main():
    config = vars(parse_args())
    args: argparse.Namespace
    rank: int
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()#WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])


    if config['arch']=='UNet':
        model = archs.__dict__[config['arch']](config['num_classes'])
    else:
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               config['deep_supervision']
                                               )
    #
    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])

    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    img_idst = glob(os.path.join('inputs', "dsb2018_96", 'images', '*' + config['img_ext']))
    img_idst = [os.path.splitext(os.path.basename(p))[0] for p in img_idst]

    # Data loading code
    img_idsv = glob(os.path.join('inputs', "val", 'images', '*' + config['img_ext']))
    img_idsv = [os.path.splitext(os.path.basename(p))[0] for p in img_idsv]

    train_img_ids = img_idst
    val_img_ids = img_idsv

    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    #数据增强：
    # 定义随机种子c

    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),

        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),#按照归一化的概率选择执行哪一个

        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ],)

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ],)

    train_dataset = Dataset(
        img_ids=train_img_ids,
        # img_dir=os.path.join('inputs', config['dataset'], 'images'),
        # mask_dir=os.path.join('inputs', config['dataset'], 'masks'),

        img_dir=os.path.join('inputs', "dsb2018_96", 'images'),
        mask_dir=os.path.join('inputs', "dsb2018_96", 'masks'),

        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],

        num_classes=config['num_classes'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', "val", 'images'),
        mask_dir=os.path.join('inputs', "val", 'masks'),

        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],

        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)#不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('recall', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_recall', []),

    ])

    best_iou = 0
    best_dice = 0
    best_dice60 = 0
    best_loss = 100
    best_loss50 = 100
    trigger = 0
    train_loss = []
    val_loss = []
    i=0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])




        print('loss %.4f - iou %.4f - dice %.4f - recall %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_recall %.4f'
              % (train_log['loss'], train_log['iou'],train_log['dice'],train_log['recall'], val_log['loss'], val_log['iou'],val_log['dice'],val_log['recall']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['recall'].append(train_log['recall'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_recall'].append(val_log['recall'])

        train_loss.append(train_log['loss'])
        val_loss.append(val_log['loss'])
        # with open("models/dsb2018_96_UNet_woDS/train_loss.txt", 'w') as train_los:
        #    train_los.write(str(train_loss))
        # with open("models/dsb2018_96_UNet_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))

        # with open("models/dsb2018_96_NestedUNet_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_NestedUNet_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))

        # with open("models/dsb2018_96_DeepLabV3Plus_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_DeepLabV3Plus_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))
        #AttU_Net
        # with open("models/dsb2018_96_SegNet_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_SegNet_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))


        # with open("models/dsb2018_96_DFANet_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_DFANet_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))

        # with open("models/dsb2018_96_TransUnet_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_TransUnet_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))

        # with open("models/dsb2018_96_GRUUNet_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_GRUUNet_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))
        # with open("models/dsb2018_96_FPN_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_FPN_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))


        # with open("models/dsb2018_96_RUNet_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_RUNet_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))


        #
        # with open("models/dsb2018_96_RUNet2_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_RUNet2_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))



        # with open("models/dsb2018_96_ResNeXtUnet18_34_woDS/train_loss.txt", 'w') as train_los:
        #     train_los.write(str(train_loss))
        # with open("models/dsb2018_96_ResNeXtUnet18_34_woDS/val_loss.txt", 'w') as val_los:
        #     val_los.write(str(val_loss))

        # os.path.join('models',config['name'], 'train_loss.txt')
        # val_loss
        with open(os.path.join('models',config['name'], 'train_loss.txt'), 'w') as train_los:
            train_los.write(str(train_loss))
        with open(os.path.join('models',config['name'], 'val_loss.txt'), 'w') as val_los:
            val_los.write(str(val_loss))

        loss(config['name'])
        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        # if val_log['iou'] > best_iou:
        #     torch.save(model.state_dict(), 'models/%s/model.pth' %
        #                config['name'])
        #     best_iou = val_log['iou']
        #     print("=> saved best model")
        #     trigger = 0


        # torch.save(model.state_dict(), f'models/%s/model{i}.pth' %
        #            config['name'])
        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_loss = val_log['loss']
            print("=> saved best model")
            trigger = 0
        if (val_log['loss'] < best_loss50) & (i>=60) :
            torch.save(model.state_dict(), 'models/%s/model2.pth' %
                       config['name'])
            best_loss50 = val_log['loss']
            print("=> saved best model2")
            trigger = 0



        i = i + 1
        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
