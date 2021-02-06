from .kon import KONDetection, KONAnnotationTransform, KON_CLASSES, KON_ROOT
from .config import *
import torch
import cv2
import numpy as np


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, boxes=None):
    height, width, _ = image.shape
    # normalize
    image = image.astype(np.float32)
    # zero padding
    if height > width:
        image_ = np.zeros([height, height, 3])
        delta_w = height - width
        left = delta_w // 2
        image_[:, left:left+width, :] = image
        offset = np.array([[ left / height, 0.,  left / height, 0.]])
        scale =  np.array([[width / height, 1., width / height, 1.]])

    elif height < width:
        image_ = np.zeros([width, width, 3])
        delta_h = width - height
        top = delta_h // 2
        image_[top:top+height, :, :] = image
        offset = np.array([[0.,    top / width, 0.,    top / width]])
        scale =  np.array([[1., height / width, 1., height / width]])

    else:
        image_ = image
        scale =  np.array([[1., 1., 1., 1.]])
        offset = np.zeros([1, 4])
    if boxes is not None:
        boxes = boxes * scale + offset

    # resize
    image_ = cv2.resize(image_, (size[1], size[0])).astype(np.float32)
    # normalize
    image_ /= 255.
    
    return image_, boxes, scale, offset


class BaseTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image, boxes, scale, offset = base_transform(image, self.size, boxes)

        return image, boxes, labels, scale, offset
