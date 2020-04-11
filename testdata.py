import os

import cv2
import numpy as np
import torch
from PIL import Image

from data import RandomFlip, RandomRotate, RandomNoise, ToTensor, my_parse_line


def test_sample(sample):
    rr = RandomRotate()
    sample = rr(sample)
    img = np.array(sample['image'])
    cv2.imshow('RandomRotate', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    rlr = RandomFlip()
    sample = rlr(sample)
    img = np.array(sample['image'])
    cv2.imshow('RandomFlipLR', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    rn = RandomNoise()
    sample['image'] = torch.from_numpy(img)
    sample = rn(sample)
    img = np.array(sample['image'])
    cv2.imshow('RandomRotate', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def load_data(phase, net, angle):
    data_file = phase + '.csv'
    data_file = os.path.join('traffic-sign', data_file)
    with open(data_file) as f:
        lines = f.readlines()[1:]
    return lines


def get_train_test_set(net, angle):
    train_set = load_data('train_label', net, angle)
    valid_set = load_data('test_label', net, angle)
    return train_set, valid_set


def main_test(args):
    train_set, test_set = get_train_test_set(args['net'], args['angle'])

    for line in train_set:
        param = my_parse_line(line)
        sample = {'image': Image.open(param[1]).convert('RGB'), 'net': 'resnet101', 'angle': 30}
        test_sample(sample)
    for line in test_set:
        param = my_parse_line(line)
        test_sample(sample)


if __name__ == '__main__':
    args = {'net': '', 'angle': 15}
    main_test(args)
