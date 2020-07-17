#!/usr/bin/python
# -*- coding: UTF-8 -*-
from random import randint
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import skimage

def get_noise_img(imagesize):
    img = np.zeros((imagesize,imagesize),dtype=np.uint8)
    for i in range(randint(3,8)):
        x=random.randint(0,imagesize)
        y=random.randint(0,imagesize)
        cv2.circle(img,(x,y),randint(1,5),(255,255,255),-1)
    return img
def get_noise_img_array(imagesize,img_num):
    img = np.zeros((imagesize,imagesize,img_num),dtype=np.uint8)
    for i in range(img_num):
        img[:,:,i] = get_noise_img(imagesize)
    return img
def get_target_img():
    image_range = 1852/2.0
    imagesize = [640,640]
    img_num = 5
    image = np.zeros((imagesize[0],imagesize[1],img_num),dtype=np.uint8)
    angle = random.randint(0,3600)/10.0
    volocity = random.randint(0,350)/10.0
    v_x = np.cos(angle*np.pi/180)*volocity
    v_y = np.sin(angle*np.pi/180)*volocity
    # print(angle, v_x, v_y)
    x=random.randint(0,640)
    y=random.randint(0,640)
    radius = random.randint(2,8)
    for num in range(5):
        img = np.zeros((imagesize[0],imagesize[1]),dtype=np.uint8)
        img_x = np.int(x+v_x*num)
        img_y = np.int(y+v_y*num)
        if max(img_x,img_y)<imagesize[0] and min(img_x,img_y)>=0:
            cv2.circle(img,(img_x,img_y),randint(1,5),(255,255,255),-1)
        image[:,:,num] = img
    return image
def get_target_img_array(imagesize,img_num):
    image = np.zeros((imagesize,imagesize,img_num),dtype=np.uint8)
    for i in range(random.randint(2,6)):
        image = image + get_target_img()
        # print(i)
    return image
def get_train_data(data_num):
    img_num = 5
    data = np.zeros((data_num,640,640,img_num),dtype=np.uint8) 
    label = np.zeros((data_num,640,640),dtype=np.uint8) 
    for i in range(data_num):
        train_image = get_target_img_array(640,5)
        train_label = train_image[:,:,0]
        train_image = train_image+get_noise_img_array(640,5)>0
        data[i,:,:,:] = train_image
        label[i,:,:] = train_label/255
    return data,label


