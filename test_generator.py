from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras.callbacks import TensorBoard




from keras_frcnn.get_train import get_data


# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = True
C.use_vertical_flips = True
C.rot_90 = True

C.model_path = True
C.num_rois = int(4)


C.network = 'vgg'
from keras_frcnn import vgg as nn


# C.base_net_weights = nn.get_weight_path()

# 加载训练集
all_imgs, classes_count, class_mapping = get_data("./train/annotation_1.csv")

# bg 
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = "test.pickle"

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

# train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
# val_imgs = [s for s in all_imgs if s['imageset'] == 'val']
# test_imgs = [s for s in all_imgs if s['imageset'] == 'test']
train_imgs = [s for s in all_imgs]


print('Num train samples {}'.format(len(train_imgs)))
# print('Num val samples {}'.format(len(val_imgs)))
# print('Num test samples {}'.format(len(test_imgs)))

# groundtruth anchor
print(nn.get_img_output_length)
data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
# data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')
# data_gen_test = data_generators.get_anchor_gt(test_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')
print("data_generators运行完成")

# X, Y, img_data = next(data_gen_train)
print("加载一个batch的数据")
