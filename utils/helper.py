import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

import cv2 
from matplotlib.patches import Circle
from scipy.misc import imsave


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))




def gen_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, '*.png')):


        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        startTime = time.clock()
        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        endTime = time.clock()
        speed_ = 1.0 / (endTime - startTime)

        yield os.path.basename(image_file), np.array(street_im), speed_



def test(sess, logits, keep_prob, image_pl, img, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    image = scipy.misc.imresize(img, image_shape)
    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return street_im, mask


def pred_single(runs_dir, img, sess, image_shape, logits, keep_prob, input_image, print_speed=False, visualize = False):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Predicting images...')
    # start epoch training timer

    image_outputs, road_mask = test(
        sess, logits, keep_prob, input_image, img, image_shape)

    counter = 0
    
    image = scipy.misc.imresize(image_outputs, (720,1280))
    # scipy.misc.imsave(os.path.join(output_dir, name), image)
    if visualize:
        img = np.uint8(image)
        plt.imshow(img)
        plt.show()

    if print_speed is True:
        counter+=1
        print("Processing file: {0:05d},\tSpeed: {1:.2f} fps".format(counter, speed_))

    return img, road_mask


def pred_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, print_speed=False, visualize = False):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Predicting images...')
    # start epoch training timer

    image_outputs = gen_output(
        sess, logits, keep_prob, input_image, data_dir, image_shape)

    counter = 0
    for name, image, speed_ in image_outputs:
        image = scipy.misc.imresize(image, (720,1280))
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        if visualize:
            img = np.uint8(image)
            plt.imshow(img)
            plt.show()

        if print_speed is True:
            counter+=1
            print("Processing file: {0:05d},\tSpeed: {1:.2f} fps".format(counter, speed_))

        # sum_time += laptime

    # pngCounter = len(glob1(data_dir,'*.png'))

    print('All augmented images are saved to: {}.'.format(output_dir))



def scale_abs(x, m = 255):
  x = np.absolute(x)
  x = np.uint8(m * x / np.max(x))
  return x 

def roi(gray, mn = 125, mx = 1200):
  m = np.copy(gray) + 1
  m[:, :mn] = 0 
  m[:, mx:] = 0 
  return m 

def save_image(img, name, i):
  path =  "output_images/" + name + str(i) + ".jpg"
  imsave(path, img)


def show_images(imgs, per_row = 3, per_col = 2, W = 10, H = 5, tdpi = 80):
      
  fig, ax = plt.subplots(per_col, per_row, figsize = (W, H), dpi = tdpi)
  ax = ax.ravel()
  
  for i in range(len(imgs)):
    img = imgs[i]
    ax[i].imshow(img)
  
  for i in range(per_row * per_col):
    ax[i].axis('off')


def show_dotted_image(this_image, points, thickness = 5, color = [255, 0, 255 ], d = 15):
  
  image = this_image.copy()
  
  cv2.line(image, points[0], points[1], color, thickness)
  cv2.line(image, points[2], points[3], color, thickness)

  fig, ax = plt.subplots(1)
  ax.set_aspect('equal')
  ax.imshow(image)

  for (x, y) in points:
    dot = Circle((x, y), d)
    ax.add_patch(dot)
  
  plt.show()
