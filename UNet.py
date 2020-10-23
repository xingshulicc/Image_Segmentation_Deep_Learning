# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Wed Oct 14 18:09:27 2020

@author: Admin
"""
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Reshape
#from keras.layers import MaxPool2D
from keras.layers import concatenate
from keras.layers import UpSampling2D
from keras.models import Model

from keras.applications.vgg16 import VGG16

def UNet(nClasses, input_height, input_width):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    input_shape = (input_height, input_width, 3)
    img_input = Input(shape = input_shape)
    bn_axis = -1
    
    vgg_streamlined = VGG16(include_top = False, 
                            weights = 'imagenet', 
                            input_tensor = img_input, 
                            pooling = None)
    assert isinstance(vgg_streamlined, Model)
    
    o = UpSampling2D(size = (2, 2))(vgg_streamlined.output)
    v1 = vgg_streamlined.get_layer(name = 'block4_pool').output
    o = concatenate([v1, o], axis = bn_axis)
    o = Conv2D(filters = 512, 
               kernel_size = (3, 3), 
               strides = (1, 1), 
               padding = 'same')(o)
    o = BatchNormalization(axis = bn_axis)(o)
#    The shape of o: 14 x 14 x 512
    
    o = UpSampling2D(size = (2, 2))(o)
    v2 = vgg_streamlined.get_layer(name = 'block3_pool').output
    o = concatenate([v2, o], axis = bn_axis)
    o = Conv2D(filters = 256, 
               kernel_size = (3, 3), 
               strides = (1, 1), 
               padding = 'same')(o)
    o = BatchNormalization(axis = bn_axis)(o)
#    The shape of o: 28 x 28 x 256
    
    o = UpSampling2D(size = (2, 2))(o)
    v3 = vgg_streamlined.get_layer(name = 'block2_pool').output
    o = concatenate([v3, o], axis = bn_axis)
    o = Conv2D(filters = 128, 
               kernel_size = (3, 3), 
               strides = (1, 1), 
               padding = 'same')(o)
    o = BatchNormalization(axis = bn_axis)(o)
#    The shape of o: 56 x 56 x 128
    
    o = UpSampling2D(size = (2, 2))(o)
    v4 = vgg_streamlined.get_layer(name = 'block1_pool').output
    o = concatenate([v4, o], axis = bn_axis)
    o = Conv2D(filters = 64, 
               kernel_size = (3, 3), 
               strides = (1, 1), 
               padding = 'same')(o)
    o = BatchNormalization(axis = bn_axis)(o)
#    The shape of o: 112 x 112 x 64
    
    o = UpSampling2D(size = (2, 2))(o)
    o = Conv2D(filters = 64, 
               kernel_size = (3, 3), 
               strides = (1, 1), 
               padding = 'same')(o)
    o = BatchNormalization(axis = bn_axis)(o)
#    The shape of o: 224 x 224 x 64
    
    o = Conv2D(filters = nClasses, 
               kernel_size = (1, 1), 
               strides = (1, 1), 
               padding = 'same')(o)
    o = BatchNormalization(axis = bn_axis)(o)
    o = Activation('relu')(o)   
#    The shape of o: 224 x 224 x nClasses
    
    o = Reshape((-1, nClasses))(o)
#    The shape of o: (input_height * input_width, nClasses)
    o = Activation('softmax')(o)
    
    model = Model(inputs = img_input, outputs = o)
    return model

if __name__ == '__main__':
    m = UNet(15, 320, 320)
    from keras.utils import plot_model
    plot_model(m, show_shapes = True, to_file = 'model_unet.png')
    print(len(m.layers))
    m.summary() 
    
    
