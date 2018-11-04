from __future__ import division
import numpy as np
import pandas as pan
import cv2
import os
import tensorflow as tf
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework import dtypes

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
main_path = "epfl_dataset/"
train_csv_file = "data_train_labelM09N08.csv"
validate_csv_file = "data_validate_labelM09N08.csv"
#test_csv_file = "data_test_labelM08N09.csv"
# test_csv_file = "data_test.csv"
mean_value_path = "mean.npy"

class ImageDataGen():
  def __init__(self, num_samples, batch_size, num_start_angles, num_angles, mode = 'training', horizontal_flip=False):

    # Init params
    self.horizontal_flip = horizontal_flip
    #self.shuffle = shuffle
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.num_start_angles = num_start_angles
    self.num_angles = num_angles

    self.filldata(mode)

  def filldata(self,mode):

    if mode == 'training':
      stream = np.array(pan.read_csv(train_csv_file))
    elif mode == 'validation':
      stream = np.array(pan.read_csv(validate_csv_file))
    else:
      raise ValueError("Invalid mode '%s'" % (mode))

    ############ NOTE USE DATASET.MAP FOR HUGE INPUT DATA INSTEAD OF CONVERTING TO NUMPY...USING THIS WAY JUST TO FIND MEAN #####

    self.image_paths = []
    self.targets = []
    self.angles = []

    for i in range(stream.shape[0]):
      self.image_paths.append(stream[i,0]) #anti_aliasing=True
      self.targets.append(stream[i,2:self.num_start_angles+2])
      self.angles.append(stream[i,1])

    self.mean_bgr = np.load(mean_value_path)

    self.image_paths = convert_to_tensor(self.image_paths, dtype = dtypes.string)
    self.targets = convert_to_tensor(np.array(self.targets), dtype = dtypes.int32)
    self.angles = convert_to_tensor(np.array(self.angles), dtype = dtypes.float32)

    data = tf.data.Dataset.from_tensor_slices((self.image_paths, self.targets, self.angles)).repeat()

    data = data.map(self._parse_function, num_parallel_calls = 8)

    if mode == 'training':
      data.shuffle(buffer_size = int(stream.shape[0]/2))

    data = data.batch(self.batch_size)

    self.data = data

  def _parse_function(self, filename, labels, angles):
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    img_decoded = tf.cast(img_decoded, tf.float32)
    img_bgr = img_decoded[:,:,::-1]
    img_centred = tf.subtract(img_bgr, self.mean_bgr)

    return img_centred, labels, angles
