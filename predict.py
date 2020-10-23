# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Oct 20 17:07:20 2020

@author: Admin
"""
import UNet
import LoadBatches
import glob
import os
import cv2
import numpy as np
import random

n_classes = 11
input_height = 320
input_width = 320

#load images path
image_path = 'single_test/image_test/'
seg_path = 'single_test/annotation_test/'
pred_path = 'single_test/model_predict/'

#color the predictive images: the order is B, G, R
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

def label2color(colors, n_classes, seg):
    seg_color = np.zeros((seg.shape[0], seg.shape[1], 3))
    for c in range(n_classes):
        seg_color[:, :, 0] += ((seg == c) * (colors[c][0])).astype('uint8')
        seg_color[:, :, 1] += ((seg == c) * (colors[c][1])).astype('uint8')
        seg_color[:, :, 2] += ((seg == c) * (colors[c][2])).astype('uint8')
        
    seg_color = seg_color.astype(np.uint8)
    
    return seg_color

# get image patches
def getcenteroffset(shape, input_height, input_width):
    assert shape[0] >= input_height and shape[1] >= input_width
    xx = int((shape[0] - input_height) / 2)
    yy = int((shape[1] - input_width) / 2)
    
    return xx, yy

#get input images and label_images
image = sorted(glob.glob(image_path + '*.jpg') + 
               glob.glob(image_path + '*.png') + 
               glob.glob(image_path + '*.jpeg'))

segmentation = sorted(glob.glob(seg_path + '*.jpg') + 
                      glob.glob(seg_path + '*.png') + 
                      glob.glob(seg_path + '*.jpeg'))

#load pre-trained model weights for predict
model = UNet.UNet(n_classes, input_height, input_width)
save_dir = os.path.join(os.getcwd(), 'Best_Performance_Model')
model_weights_name = 'keras_trained_model_weights.h5'
model_weights_path = os.path.join(save_dir, model_weights_name)
model.load_weights(model_weights_path)
#the shape of the model output is (input_height * input_width, n_classes)

im = cv2.imread(image, 1)
xx, yy = getcenteroffset(im.shape, input_height, input_width)
im = im[xx:xx + input_height, yy:yy + input_width, :]

seg = cv2.imread(segmentation, 0)
seg = seg[xx:xx + input_height, yy:yy + input_width]

pr = model.predict(np.expand_dims(LoadBatches.getImageArr(im), 0))[0]
pr = pr.reshape((input_height, input_width, n_classes)).argmax(axis = 2)
#the shape of pr is: input_height, input_width

cv2.imshow('input_image', im)
cv2.imshow('predictive_segmentation', label2color(colors, n_classes, pr))
cv2.imshow('ground_truth_segmentation', label2color(colors, n_classes, seg))
cv2.waitKey()


