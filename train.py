# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Oct 20 13:44:36 2020

@author: Admin
"""
import LoadBatches
import UNet
import os

from keras.optimizers import SGD
#from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from learning_rate import choose

#load image data path
train_images_path = 'dataset1/images_prepped_train/'
train_segs_path = 'dataset1/annotations_prepped_train/'

val_images_path = 'dataset1/images_prepped_test/'
val_segs_path = 'dataset1/annotations_prepped_test/'

nb_train_samples = 367
nb_validation_samples = 101

#set model hyper-parameters
batch_size = 4
n_classes = 11
epochs = 500
input_height = 320
input_width = 320
lr = 0.001

#load and compile model 
optimizer = SGD(lr = lr, momentum = 0.9, nesterov = True)
#optimizer = Adam(lr = lr, amsgrad = True)

model = UNet.UNet(n_classes, input_height, input_width)
model.compile(optimizer = optimizer, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

#load training and test image data (training and test metrics)
G = LoadBatches.imageSegmentationGenerator(train_images_path, 
                                           train_segs_path, 
                                           batch_size, 
                                           n_classes, 
                                           input_height, 
                                           input_width)
G_test = LoadBatches.imageSegmentationGenerator(val_images_path, 
                                                val_segs_path, 
                                                batch_size, 
                                                n_classes, 
                                                input_height, 
                                                input_width)

#set callbacks
lr_monitorable = True
lr_reduce = choose(lr_monitorable = lr_monitorable)

save_dir = os.path.join(os.getcwd(), 'Best_Performance_Model')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_weights_name = 'keras_trained_model_weights.h5'
save_path = os.path.join(save_dir, model_weights_name)
checkpoint = ModelCheckpoint(filepath = save_path, 
                             monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = 'max', 
                             save_weights_only = True)

callbacks = [lr_reduce, checkpoint]

#train and test model
hist = model.fit_generator(generator = G, 
                           steps_per_epoch = nb_train_samples // batch_size, 
                           epochs = epochs, 
                           callbacks = callbacks, 
                           validation_data = G_test, 
                           validation_steps = nb_validation_samples // batch_size, 
                           shuffle = True)

#store training results
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print val_acc and stored into val_acc.txt
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()



#please note that: the size of input images should be larger than 320 * 320 (input_height, input_width)
#TO DO:
#reise input images and segementations in advance to avoid this problem
