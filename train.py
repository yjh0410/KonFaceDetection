from data import *
from utils import SSDAugmentation, MAPEvaluator
import tools
import torch.backends.cudnn as cudnn
import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

import matplotlib.pyplot as plt



def parse_args():
    parser = argparse.ArgumentParser(description='Kon-Face Detection')
    parser.add_argument('-v', '--version', default='CenterFace',
                        help='CenterFace')
    parser.add_argument('-d', '--dataset', default='widerface',
                        help='widerface dataset')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=8, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=5,
                        help='The upper bound of warm-up')
    parser.add_argument('--eval_epoch', type=int, default=10,
                        help='eval epoch')
    parser.add_argument('-r', '--pretrained', action='store_true', default=False, 
                        help='use model pre-trained on WiderFace')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')

    return parser.parse_args()


def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    hr = False  
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True

    cfg = WF_config

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # multi scale
    if args.multi_scale:
        print('Let us use the multi-scale trick.')
        train_size = [800, 800]
        val_size = [640, 640]
    else:
        train_size = [640, 640]
        val_size = [640, 640]

    # we use our ship dataset
    num_classes = 5
    data_dir = KON_ROOT
    train_dataset = KONDetection(root=data_dir, 
                            img_size=train_size[0],
                            transform=SSDAugmentation(train_size),
                            base_transform=BaseTransform(train_size),
                            mosaic=args.mosaic
                            )

    val_dataset = KONDetection(root=data_dir, 
                               img_size=val_size[0],
                               transform=BaseTransform(train_size),
                                )

    evaluator = MAPEvaluator(device=device,
                             dataset=val_dataset,
                             classname=KON_CLASSES,
                             name='kon',
                             display=True
                             )

    # build model
    if args.version == 'CenterFace':
        from models.centerface import CenterFace
        pretrained_path = 'weights/pretrained/CenterFace.pth'

        net = CenterFace(device, input_size=train_size, num_classes=num_classes, trainable=True)
        print('Let us train CenterFace......')

    else:
        print('Unknown version !!!')
        exit()

    model = net
    model.to(device)
    # keep training
    if args.pretrained:
        print('use model pretrained on widerface: %s' % (pretrained_path))
        model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/widerface/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    print("----------------------------------------Face Detection--------------------------------------------")

    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)

    print("----------------------------------------------------------")
    print('Training on:', train_dataset.name)
    print('The dataset size:', len(train_dataset))
    print('Initial learning rate: ', args.lr)
    print("----------------------------------------------------------")

    epoch_size = len(train_dataset) // args.batch_size
    max_epoch = cfg['max_epoch']

    data_loader = data.DataLoader(train_dataset, 
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    t0 = time.time()

    # start training
    for epoch in range(max_epoch):      
        # use step lr
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets) in enumerate(data_loader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
                    
            # to device
            images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                size = random.randint(10, 25) * 32
                input_size = [size, size]
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=False)

            # make train label
            targets = [label.tolist() for label in targets]
            # vis_data(images, targets, train_Size)
            targets = tools.gt_creator(train_size, net.stride, num_classes, targets)
            # vis_heatmap(targets)
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            cls_loss, txty_loss, twth_loss, total_loss = model(images, target=targets)
                     
            # backprop and update
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('txty loss',  txty_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('twth loss',  twth_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('total loss', total_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: cls %.2f || txty %.2f || twth %.2f ||total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            cls_loss.item(), txty_loss.item(), twth_loss.item(), total_loss.item(), train_size[0], t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        # I don't build test dataset ...
        # if (epoch + 1) % args.eval_epoch == 0:
        #     model.trainable = False
        #     model.set_grid(val_size)
        #     model.eval()

        #     # evaluate
        #     evaluator.evaluate(model)
        #     mAP = evaluator.mAP
            
        #     # convert to training mode.
        #     model.trainable = True
        #     model.set_grid(train_size)
        #     model.train()

        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth')  
                    )



def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    cv2.imwrite('1.jpg', img)

    img_ = cv2.imread('1.jpg')
    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size[0]
        ymin *= input_size[1]
        xmax *= input_size[0]
        ymax *= input_size[1]
        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img_)
    cv2.waitKey(0)


def vis_heatmap(targets):
    # vis heatmap
    heatmap = targets[0, :, 0].reshape(160, 160)
    heatmap = cv2.resize(heatmap, (640, 640))
    cv2.imshow('ss',heatmap)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()