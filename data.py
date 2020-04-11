import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import math
import numpy as np
import random
from torchvision import transforms


def erase(img, i, j, h, w, v, inplace=False):
    """ Erase the input Tensor Image with given value.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        Tensor Image: Erased image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.clone()

    img[:, i:i + h, j:j + w] = v
    return img


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def my_parse_line(line):
    """ 按行读取数据
        Args:
             line: train.txt/test.txt的一行输入，用于解读图片、人脸框和关键点
        Returns:
            img_name, rect, landmarks
    """
    line_parts = line.strip().split(',')
    id = line_parts[0]
    path = line_parts[1]
    cate = line_parts[2]
    return id, path, cate


class Normalize(object):
    """
        Resize to train_boarder x train_boarder.
        Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        image, net = sample['image'], sample['net']
        if net == '':
            train_boarder = 112
        else:
            train_boarder = 224
        # Resize image
        image_resize = np.asarray(image.resize((train_boarder, train_boarder), Image.BILINEAR), dtype=np.float32)
        # Normalization
        image = channel_norm(image_resize)
        return {'image': image,
                'net': net}


class RandomRotate(object):
    """
        Rotate the picture small angle
    """
    def __call__(self, sample):
        image, net, angle = sample['image'], sample['net'], sample['angle']
        a0 = (random.random()-0.5) * min(angle, 10)
        a1, a2 = a0, a0 * math.pi / 180
        ox, oy = image.width // 2, image.height // 2
        image = image.rotate(-a1, Image.BILINEAR, expand=0)

        return {'image': image,
                'net': net}


class RandomFlip(object):
    """
        Randomly flip left and right
    """
    def __call__(self, sample):
        image, net = sample['image'], sample['net']
        # Flip image randomly
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': image,
                'net': net}


class RandomNoise(object):
    """
        随机产生噪点
    """
    def __call__(self, sample):
        image, net = sample['image'], sample['net']
        # Flip image randomly
        img_h, img_w, img_c = image.shape
        for i in range(random.randint(100, 500)):
            h = random.randint(0, img_h-5)
            w = random.randint(0, img_w-5)
            image[h, w, :] = 255
        return {'image': image,
                'net': net}

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """

    def __call__(self, sample):
        image, net = sample['image'], sample['net']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if net == '':
            image = np.expand_dims(image, axis=0)
        else:
            image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'net': net}


def compute_iou(bbox1, bbox2):
    """
    Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2]
    :param bbox1: (tensor) bounding boxes, size [N,4]
    :param bbox2: (tensor) bounding boxes, size [M,4]
    :return:
    """
    box1 = torch.FloatTensor([bbox1])  # [N,4], 4=[x1,y1,x2,y2]
    box2 = torch.FloatTensor([bbox2])  # [M,4], 4=[x1,y1,x2,y2]
    N = box1.size(0)
    M = box2.size(0)

    tl = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )
    br = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = br - tl  # [N,M,2]
    wh[(wh < 0).detach()] = 0
    # wh[wh<0] = 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


def generate_error_data(rect, w, h, iou=0.3, error_rate=1/4.0):
    framex, framey = rect[2]-rect[0], rect[3]-rect[1]
    if random.random() > error_rate:
        x, y = random.randint(-framex, framex), random.randint(-framey, framey)
        while 0 >= rect[0]+x or 0 >= rect[1]+y or rect[2] > w or rect[3] > h \
                or iou > compute_iou([rect[0]+x, rect[1]+y, rect[2]+x, rect[3]+y], rect):
            x, y = random.randint(-framex, framex), random.randint(-framey, framey)
        return [rect[0]+x, rect[1]+y, rect[2]+x, rect[3]+y], 0
    return rect, 1


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, phase, net, angle, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.phase = phase
        self.transform = transform
        self.net = net
        self.angle = angle

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        id, path, cate = my_parse_line(self.lines[idx])
        # image
        img = Image.open(path).convert('RGB')
        sample = {'image': img, 'net': self.net, 'angle': self.angle}
        sample = self.transform(sample)
        sample['cate'] = int(cate)
        if self.phase != 'train':
            sample['path'] = path
            sample['id'] = id
        return sample


def load_data(phase, net, angle):
    data_file = phase + '.csv'
    data_file = os.path.join('traffic-sign', data_file)
    with open(data_file) as f:
        lines = f.readlines()[1:]
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            RandomRotate(),
            RandomFlip(),
            RandomNoise(),
            Normalize(),  # do channel normalization
            ToTensor()   # convert to torch type: NxCxHxW
        ]
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase, net, angle, transform=tsfm)
    return data_set


def get_train_test_set(net, angle):
    train_set = load_data('train_label', net, angle)
    valid_set = load_data('test_label', net, angle)
    return train_set, valid_set

