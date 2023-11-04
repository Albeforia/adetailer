#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhaoyang Sun

import os

import cv2
import numpy as np
import torchvision


def tensor2img(img):
    img = img.cpu()[0].float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.astype(np.uint8)


def save_imgs(imgs, names, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for img, name in zip(imgs, names):
        img = tensor2img(img)
        img = np.array(img)
        cv2.imwrite(os.path.join(path, name + '.jpg'), img)


class Saver():
    def __init__(self, opts):
        self.result_dir = opts.result_dir
        # make directory
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    # save test pair images
    def write_test_pair_img(self, iter, model, data):
        root = os.path.join(self.result_dir, 'test_pair')
        if not os.path.exists(root):
            os.makedirs(root)
        test_pair_img = model.test_pair(data)
        img_filename = '%s/gen_%05d.jpg' % (root, iter)
        torchvision.utils.save_image(test_pair_img / 2 + 0.5, img_filename, nrow=1)

    def write_test_single_img(self, iter, model, data):
        size = [int(tensor.item()) for tensor in data['original_size']]

        n1 = os.path.splitext(os.path.basename(data['non_makeup_path'][0]))[0]
        n2 = os.path.splitext(os.path.basename(data['makeup_path'][0]))[0]
        root = self.result_dir
        if not os.path.exists(root):
            os.makedirs(root)
        output_img = model.test_single(data)
        # img_filename = '%s/%s_%s.jpg' % (root, n1, n2)
        img_filename1 = '%s/out0.png' % (root)
        img_filename2 = '%s/out.png' % (root)
        torchvision.utils.save_image(output_img / 2 + 0.5, img_filename1)
        tmp_img = cv2.imread(img_filename1)
        tmp_img = cv2.resize(tmp_img, size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_filename2, tmp_img)
