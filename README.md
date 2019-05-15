# Self-Driving-Guidance
##### *Nuo Xu, Yao Li, Sikao Xiao*

This is a course project for **Deep Learning**, 2019 @ **Johns Hopkins University.** This repository presents a system approach to generate driving strategies only based on RGB camera input. The whole system is divided into road segmentation, car-detection and depth estimation. 



## Reference
**Depth Estimation** based on RGB input: https://github.com/mrharicot/monodepth

**Road Segmentation:** VGG16 + FCN. We applied pre-trained model and fine-tuning with KITTI.

**Car Detection:** Car Detection is realized by YoLo v3, which can be referred from https://github.com/SKYSHAW1996/keras-yolo3-improved

## Demo

Bothe videos are downloaded from Youtube

![](https://github.com/NormXU/Self-Driving-Dev/raw/master/Doc/demo2.gif)

# Cherish our lives !
![](https://github.com/NormXU/Self-Driving-Dev/raw/master/Doc/demo3.gif)

