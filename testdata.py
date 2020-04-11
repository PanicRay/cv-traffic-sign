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


def load_data(phase):
    data_file = phase + '.csv'
    data_file = os.path.join('traffic-sign', data_file)
    with open(data_file) as f:
        lines = f.readlines()[1:]
    return lines


def get_train_test_set():
    train_set = load_data('train_label')
    valid_set = load_data('test_label')
    return train_set, valid_set


def main_test(args):
    train_set, test_set = get_train_test_set()

    for line in train_set:
        param = my_parse_line(line)
        sample = {'image': Image.open(param[1]).convert('RGB'), 'net': 'resnet101', 'angle': args['angle']}
        test_sample(sample)
    for line in test_set:
        param = my_parse_line(line)
        sample = {'image': Image.open(param[1]).convert('RGB'), 'net': 'resnet101', 'angle': args['angle']}
        test_sample(sample)


if __name__ == '__main__':
    args = {'net': '', 'angle': 15}
    '''
    from collections import Counter
    train_set = load_data('train_label')
    ts = [my_parse_line(i) for i in train_set]
    tes = [my_parse_line(i) for i in test_set]
    cate_tr = [i[2] for i in ts]
    cate_te = [i[2] for i in tes]
    print(Counter(cate_tr))
    Counter({'22': 375, '32': 316, '38': 285, '61': 282, '40': 242, '19': 231, '53': 199, '39': 196, '7': 157, '41': 148, '47': 147, '28': 125, '54': 118, '1': 110, '37': 98, '56': 95, '13': 90, '18': 81, '17': 79, '57': 78, '45': 74, '31': 63, '35': 60, '24': 48, '44': 48, '34': 46, '46': 44, '14': 43, '20': 42, '25': 42, '59': 42, '21': 40, '30': 37, '42': 35, '29': 33, '43': 30, '8': 27, '51': 27, '52': 27, '10': 21, '6': 18, '9': 18, '12': 18, '27': 18, '36': 18, '0': 15, '3': 15, '4': 15, '23': 15, '50': 15, '58': 15, '2': 13, '33': 12, '49': 12, '55': 12, '5': 11, '48': 11, '15': 9, '16': 9, '60': 9, '11': 7, '26': 6})
    
    '''
    main_test(args)