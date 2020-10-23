# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Oct 13 11:05:30 2020

@author: Admin
"""
import cv2
import os
import numpy as np

img_path = os.path.join(os.getcwd(), 'car_label.png')
img_bgr = cv2.imread(img_path, 1)

'''
colormap:
    i:0, r:0, g:0, b:0
    i:1, r:128, g:0, b:0
    i:2, r:0, g:128, b:0
    i:3, r:128, g:128, b:0
    i:4, r:0, g:0, b:128
    i:5, r:128, g:0, b:128
    i:6, r:0, g:128, b:128
    i:7, r:128, g:128, b:128
    i:8, r:64, g:0, b:0
    i:9, r:192, g:0, b:0
    i:10, r:64, g:128, b:0
    i:11, r:192, g:128, b:0
    i:12, r:64, g:0, b:128
    i:13, r:192, g:0, b:128
    i:14, r:64, g:128, b:128
    i:15, r:192, g:128, b:128
    ......

colormap to gray: G = 0.299 * R + 0.587 * G + 0.114 * B + 0.5
    i:0, r:0, g:0, b:0 -> 0
    i:1, r:128, g:0, b:0 -> 38
    i:2, r:0, g:128, b:0 -> 75
    i:3, r:128, g:128, b:0 -> 113
    i:4, r:0, g:0, b:128 -> 15
    i:5, r:128, g:0, b:128 -> 53
    i:6, r:0, g:128, b:128-> 90
    i:7, r:128, g:128, b:128 -> 128
    i:8, r:64, g:0, b:0 -> 19
    i:9, r:192, g:0, b:0 -> 57
    i:10, r:64, g:128, b:0 -> 94
    i:11, r:192, g:128, b:0 -> 133
    i:12, r:64, g:0, b:128 -> 34
    i:13, r:192, g:0, b:128 -> 72
    i:14, r:64, g:128, b:128 -> 109
    i:15, r:192, g:128, b:128 -> 147
    ......

'''

def color2label(input_image, label_list):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_height = input_image.shape[0]
    input_width = input_image.shape[1]
    n_labels = len(label_list)
    mask_images = np.zeros(shape = (input_height, input_width, n_labels), dtype = int)
    label_image = np.zeros_like(input_image)
    for i in range(n_labels):
        mask_images[:, :, i] = (input_image == label_list[i]) * (i + 1)
    label_image = np.sum(mask_images, axis = -1)
    
    return label_image


#before generate label images, we should confirm the number of classes
label_list = [38, 75, 113]
car_output_image = color2label(img_bgr, label_list)
cv2.imwrite('car_indices.png', car_output_image)
label_image_path = os.path.join(os.getcwd(), 'car_indices.png')
car_output_image = cv2.imread(label_image_path, 0)


cv2.imshow('car_label_indices', car_output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


