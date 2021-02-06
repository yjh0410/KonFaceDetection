import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d, SPP, DeConv2d
from backbone import *
import numpy as np
import tools

class CenterFace(nn.Module):
    def __init__(self, device, input_size=None, num_classes=None, trainable=False, conf_thresh=0.3, nms_thresh=0.3, topk=200, use_nms=False):
        super(CenterFace, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_nms = use_nms
        self.stride = 4
        self.topk = topk
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[0], input_size[1], input_size[0], input_size[1]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

        # backbone
        self.backbone = resnet18(pretrained=trainable)

        # neck: deconv + FPN
        self.deconv5 = DeConv2d(512, 256, ksize=2, stride=2) # 32 -> 16
        self.deconv4 = DeConv2d(256, 128, ksize=2, stride=2) # 16 -> 8
        self.deconv3 = DeConv2d(128,  64, ksize=2, stride=2) #  8 -> 4

        self.smooth5 = Conv2d(512, 512, ksize=3, padding=1)
        self.smooth4 = Conv2d(256, 256, ksize=3, padding=1)
        self.smooth3 = Conv2d(128, 128, ksize=3, padding=1)
        self.smooth2 = Conv2d( 64,  64, ksize=3, padding=1)

        # head: cls, txty, twth
        self.cls_pred = nn.Sequential(
            Conv2d(64, 32, ksize=3, padding=1, leakyReLU=True),
            nn.Conv2d(32, self.num_classes, kernel_size=1)
        )

        self.txty_pred = nn.Sequential(
            Conv2d(64, 32, ksize=3, padding=1, leakyReLU=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )
       
        self.twth_pred = nn.Sequential(
            Conv2d(64, 32, ksize=3, padding=1, leakyReLU=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )


    def create_grid(self, input_size):
        w, h = input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
        return grid_xy

    def set_grid(self, input_size):
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[0], input_size[1], input_size[0], input_size[1]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_boxes(self, pred):
        """
        input box :  [delta_x, delta_y, sqrt(w), sqrt(h)]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        pred[:, :, :2] = (torch.sigmoid(pred[:, :, :2]) + self.grid_cell) * self.stride
        pred[:, :, 2:] = (torch.exp(pred[:, :, 2:])) * self.stride

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
        
        return output

    def _gather_feat(self, feat, ind, mask=None):
        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _topk(self, scores):
        B, C, H, W = scores.size()
        
        topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), self.topk)

        topk_inds = topk_inds % (H * W)
        
        topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), self.topk)
        topk_clses = (topk_ind / self.topk).int()
        topk_inds = self._gather_feat(topk_inds.view(B, -1, 1), topk_ind).view(B, self.topk)

        return topk_score, topk_inds, topk_clses

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def forward(self, x, target=None):
        # backbone
        c2, c3, c4, c5 = self.backbone(x)
        B = c2.size(0)

        # FPN deconv
        c5 = self.smooth5(c5)
        c4 = self.smooth4(c4 + self.deconv5(c5))
        c3 = self.smooth3(c3 + self.deconv4(c4))
        c2 = self.smooth2(c2 + self.deconv3(c3))

        # head
        cls_pred = self.cls_pred(c2)
        txty_pred = self.txty_pred(c2)
        twth_pred = self.twth_pred(c2)

        # train
        if self.trainable:
            # [B, H*W, num_classes]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            # [B, H*W, 2]
            txty_pred = txty_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
            # [B, H*W, 2]
            twth_pred = twth_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)

            # compute loss
            cls_loss, txty_loss, twth_loss, total_loss = tools.loss(pred_cls=cls_pred, 
                                                                    pred_txty=txty_pred, 
                                                                    pred_twth=twth_pred, 
                                                                    label=target, 
                                                                    num_classes=self.num_classes
                                                                    )

            return cls_loss, txty_loss, twth_loss, total_loss       

        # test
        else:
            with torch.no_grad():
                # batch_size = 1
                cls_pred = torch.sigmoid(cls_pred)              
                # simple nms
                hmax = F.max_pool2d(cls_pred, kernel_size=5, padding=2, stride=1)
                keep = (hmax == cls_pred).float()
                cls_pred *= keep

                # decode box
                txtytwth_pred = torch.cat([txty_pred, twth_pred], dim=1).permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                # [B, H*W, 4] -> [H*W, 4]
                bbox_pred = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)

                # topk
                topk_scores, topk_inds, topk_clses = self._topk(cls_pred)

                topk_scores = topk_scores[0].cpu().numpy()
                topk_ind = topk_clses[0].cpu().numpy()
                topk_bbox_pred = bbox_pred[topk_inds[0]].cpu().numpy()

                if self.use_nms:
                    # nms
                    keep = np.zeros(len(topk_bbox_pred), dtype=np.int)
                    for i in range(self.num_classes):
                        inds = np.where(topk_ind == i)[0]
                        if len(inds) == 0:
                            continue
                        c_bboxes = topk_bbox_pred[inds]
                        c_scores = topk_scores[inds]
                        c_keep = self.nms(c_bboxes, c_scores)
                        keep[inds[c_keep]] = 1

                    keep = np.where(keep > 0)
                    topk_bbox_pred = topk_bbox_pred[keep]
                    topk_scores = topk_scores[keep]
                    topk_ind = topk_ind[keep]

                return topk_bbox_pred, topk_scores, topk_ind
