# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

from utils.monodepth_model import *
from utils.monodepth_dataloader import *
from utils.average_gradients import *

"""
parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
# parser.add_argumenimage_path',       type=str,   help='path to the image', required=True)
# parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', default='model/Depth/model_kitti')
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)
parser.add_argument('--path', type = str)

args = parser.parse_args()
"""

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def mono_init():
    params = monodepth_parameters(
                encoder='vgg',
                height=256,
                width=512,
                batch_size=2,
                num_threads=1,
                num_epochs=1,
                do_stereo=False,
                wrap_mode="border",
                use_deconv=False,
                alpha_image_loss=0,
                disp_gradient_loss_weight=0,
                lr_loss_weight=0,
                full_summary=False)
    
    input_height = 256
    input_width = 512
    
    
    left  = tf.placeholder(tf.float32, [2, input_height, input_width, 3])
    
    model = MonodepthModel(params, "test", left, None)


    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess_tmp = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess_tmp.run(tf.global_variables_initializer())
    sess_tmp.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess_tmp, coord=coordinator)

    # RESTORE
    checkpoint_path = 'model/Depth/model_kitti'
    restore_path = checkpoint_path.split(".")[0]
    train_saver.restore(sess_tmp, restore_path)
    
    
    
    return sess_tmp, model, left



def mono_sigle(img,model,sess, left):
    
    input_height = 256
    input_width = 512



    
    input_image = img
    original_height, original_width, num_channels = input_image.shape
    original_size = [original_height, original_width]
    
    input_image = scipy.misc.imresize(input_image, [input_height, input_width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)
    
    
    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    plt.imshow(disp_to_img, cmap='gray')
    plt.show()
    #plt.imsave(disp_to_img, cmap='gray')
    print('done!')



