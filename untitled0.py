# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:10:56 2019

@author: Nuo Xu
"""
from glob import glob
import os
import os.path
import cv2
# %%
def DistanceCompute(depthPath, position, idx, showFlag = True):
    depth = cv2.imread(depthPath,0)
    for i in range(len(position[idx])):
        top = position[idx][i][0]
        left = position[idx][i][1]
        bottom = position[idx][i][2]
        right = position[idx][i][3]
        depth = cv2.rectangle(depth,(top,left),(bottom,right),(255,255,0),3)
    if showFlag:
        plt.imshow(depth,cmap='gray')
        plt.show()
    
    
Depthdata_path = 'yolo_util/crops2/depth/'
depthimgs = glob(os.path.join(Depthdata_path, "*.png"))
for i in range(len(depthimgs)):
    depthPath = depthimgs[i]
    DistanceCompute(depthPath, results, i, showFlag = True)

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