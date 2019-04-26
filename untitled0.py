# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:10:56 2019

@author: Nuo Xu
"""
from glob import glob
import os
import os.path
import cv2
import sys
import argparse
from yolo_util.yolo import YOLO, detect_video
from PIL import Image
from yolo_util.yolo_video import *
# %%
def DistanceCompute(depthPath, position, idx, threshold = 90, showFlag = True):
    depthOrigin = cv2.imread(depthPath,0)
    for i in range(len(position[idx])):
        top = position[idx][i][0]
        left = position[idx][i][1]
        bottom = position[idx][i][2]
        right = position[idx][i][3]
        crop_Image = depthOrigin[left:right,top:bottom]
        depth_distance = int(crop_Image.mean())
        #print(depth_distance)
        
        
        depth = cv2.rectangle(depthOrigin,(top,left),(bottom,right),(255,255,0),3)
         
        #plt.imsave(os.path.join('./data/crops2/depthCrop\', "{}_disp.png".format(i)), depth, cmap='gray')
        if showFlag and depth_distance > threshold:
            plt.imshow(depth,cmap='gray')
            plt.show()
    
    
# %%
import sys
import argparse
from yolo_util.yolo import YOLO, detect_video
from PIL import Image
from yolo_util.yolo_video import *

results = detect_multiple(YOLO(), 'yolo_util/crops2/original/', "")
#%%
img_path = 'yolo_util/crops2/original/'
imgs = glob(os.path.join(img_path, "*.png"))
results[0]
#%%
import pickle
f = open('yolo_pos_crash.pckl', 'rb')
obj = pickle.load(f)
f.close()
#%%
Depthdata_path = 'yolo_util/crop3/original/'
#Depthdata_path = 'yolo_util/crops2/original/'
depthimgs = glob(os.path.join(Depthdata_path, "*.png"))
for i in range(len(depthimgs)):
    depthPath = depthimgs[i]
    DistanceCompute(depthPath, obj, i, showFlag = True)