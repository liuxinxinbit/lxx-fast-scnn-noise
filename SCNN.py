import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation,concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import LeakyReLU
import os
import random
import time
import wget
import tarfile
import numpy as np
import cv2
from PIL import Image
import os 
import sys
import glob
import json
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz
from tensorflow import keras
from layers import pyramid_pooling, bottleneck
from virtual_data import get_train_data
class SCNN:
    def __init__(self, train_stage=1,use_cpu=False, print_summary=False):
        self.build(use_cpu=use_cpu, print_summary=print_summary)

    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='model.h5'):
        self.model.load_weights(file_path)
    
    def BatchGenerator(self,batch_size=8, image_size=(640, 640, 5), labels=1):#500, 375
        while True:
            images,truths = get_train_data(image_size[2])
            yield images, truths
    def BatchEvaluate(self,batch_size=8, image_size=(640, 640, 5), labels=3):#500, 375
        images, truths = get_train_data(image_size[2])
        score = self.model.evaluate(images, truths, verbose=0)
        print(self.model.metrics_names)
        return score              
    def train(self, epochs=10, steps_per_epoch=50,batch_size=32):
        batch_generator = self.BatchGenerator(batch_size=batch_size)
        self.model.fit_generator(batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def build_conv2D_block(self, inputs, filters, kernel_size, strides):
        conv2d = Conv2D(filters = filters, kernel_size=kernel_size,strides=strides, padding='same', use_bias=True,bias_initializer='zeros')(inputs)
        conv2d = BatchNormalization()(conv2d)
        conv2d_output = Activation(LeakyReLU(alpha=0.1))(conv2d)
        return conv2d_output

    def build_conv2Dtranspose_block(self, inputs, filters, kernel_size, strides):
        conv2d = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=True,bias_initializer='zeros', padding='same')(inputs)
        conv2d = BatchNormalization()(conv2d)
        conv2d_deconv = Activation(LeakyReLU(alpha=0.1))(conv2d)
        return conv2d_deconv

    def build_DepthwiseConv2D_block(self, inputs, filters):
        Depthwiseconv2d = keras.layers.DepthwiseConv2D(filters, padding="same")(inputs)
        Depthwiseconv2d = BatchNormalization()(Depthwiseconv2d)
        Depthwiseconv2d = Activation(LeakyReLU(alpha=0.1))(Depthwiseconv2d)
        return Depthwiseconv2d

    def build_SeparableConv2D_block(self, inputs, filters,kernel_size,strides):
        Separableconv2d = keras.layers.SeparableConv2D(filters,kernel_size,strides, padding="same")(inputs)
        Separableconv2d = BatchNormalization()(Separableconv2d)
        Separableconv2d = Activation(LeakyReLU(alpha=0.1))(Separableconv2d)
        return Separableconv2d

    def my_loss_error(self, y_true, y_pred):
        return K.sum((K.abs(y_pred - y_true)))

    def build(self, use_cpu=False, print_summary=False, num_classes=1, sub_region_sizes=[1, 2, 3, 6], expansion_factor=6):
          
        input_tensor = keras.layers.Input((640, 640, 5))
        
        learning_to_down_sample1   = self.build_conv2D_block(input_tensor,filters = 32,kernel_size=3,strides=2)

        learning_to_down_sample2   = self.build_conv2D_block(learning_to_down_sample1,filters = 48,kernel_size=3,strides=2)

        learning_to_down_sample3   = self.build_conv2D_block(learning_to_down_sample2,filters = 64,kernel_size=3,strides=2)

        skip_connection1 = learning_to_down_sample1
        skip_connection2 = learning_to_down_sample2
        skip_connection3 = learning_to_down_sample3
        
        # Global feature extractor

        global_feature_extractor1 = bottleneck(learning_to_down_sample3,
                                          64, 2, expansion_factor)

        global_feature_extractor = bottleneck(global_feature_extractor1,
                                          64, 1, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          64, 1, expansion_factor)

        global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 2, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 1, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          96, 1, expansion_factor)

        global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
        global_feature_extractor = bottleneck(global_feature_extractor,
                                          128, 1, expansion_factor)
        global_feature_extractor = pyramid_pooling(global_feature_extractor,
                                               sub_region_sizes)

        # Feature fusion

        conv2d_deconv5_1 = self.build_conv2D_block(global_feature_extractor,filters = 196,kernel_size=3,strides=1)
        conv2d_deconv4   = self.build_conv2Dtranspose_block(conv2d_deconv5_1, filters=196, kernel_size=4, strides=2)

        #Concat4
        Concat_concat4 = concatenate([conv2d_deconv4, global_feature_extractor1] , axis=-1)

        conv2d_deconv4_1 = self.build_conv2D_block(Concat_concat4,filters = 128,kernel_size=3,strides=1)
        conv2d_deconv3   = self.build_conv2Dtranspose_block(conv2d_deconv4_1, filters=128, kernel_size=4, strides=2)
 
        #Concat4
        feature_fusion_main_branch = concatenate([conv2d_deconv3, skip_connection3] , axis=-1)

        feature_fusion_main_branch = self.build_DepthwiseConv2D_block(feature_fusion_main_branch,filters = 3)

        feature_fusion_main_branch = keras.layers.Conv2D(
            128, 1, 1, padding="same")(feature_fusion_main_branch)
        feature_fusion_main_branch = keras.layers.BatchNormalization()(
            feature_fusion_main_branch)

        feature_fusion_skip_connection = keras.layers.Conv2D(
            128, 1, 1, padding="same")(skip_connection3)
        feature_fusion_skip_connection = keras.layers.BatchNormalization()(
            feature_fusion_skip_connection)

        feature_fusion = feature_fusion_main_branch + feature_fusion_skip_connection

        # Classifier
        
        classifier = self.build_SeparableConv2D_block( feature_fusion, 128,3,1)
        classifier = self.build_SeparableConv2D_block( classifier, 128,3,1)
        classifier   = self.build_conv2D_block(classifier,filters = 3,kernel_size=1,strides=1)
        #Concat4

        conv2d_deconv3_1 = self.build_conv2D_block(classifier,filters = 96,kernel_size=3,strides=1)
        conv2d_deconv2   = self.build_conv2Dtranspose_block(conv2d_deconv3_1, filters=96, kernel_size=4, strides=2)
        conv2d_deconv2 = concatenate([conv2d_deconv2, skip_connection2] , axis=-1)

        conv2d_deconv2_1 = self.build_conv2D_block(conv2d_deconv2,filters = 64,kernel_size=3,strides=1)
        conv2d_deconv1   = self.build_conv2Dtranspose_block(conv2d_deconv2_1, filters=64, kernel_size=4, strides=2)
        #Concat4
        conv2d_deconv3 = concatenate([conv2d_deconv1, skip_connection1] , axis=-1)

        conv2d_deconv1_1 = self.build_conv2D_block(conv2d_deconv3,filters = 24,kernel_size=3,strides=1)
        conv2d_deconv0   = self.build_conv2Dtranspose_block(conv2d_deconv1_1, filters=24, kernel_size=4, strides=2)

        output_tensor = Conv2DTranspose(filters=num_classes, kernel_size=1, strides=1, activation='sigmoid', padding='same', name='output')(conv2d_deconv0)
        
        self.model = keras.models.Model(input_tensor, output_tensor)
        # ~ parallel_model = multi_gpu_model(self.model, gpus=1)
        self.model.compile(optimizer='adam',loss=self.my_loss_error,metrics=['accuracy', 'mse'])
