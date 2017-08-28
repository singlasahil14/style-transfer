import tensorflow as tf
from utils import *
from loss_network_factory import *
from argparse import ArgumentParser

image_size = 224

import numpy as np
import os
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib
from nets import inception, nets_factory
from tensorflow.contrib import slim

def main():
  parser = ArgumentParser()
  parser.add_argument('--network', default='vgg-16', choices=['vgg-16', 'vgg-19', 'inception-v1', 'inception-v2', 'inception-v3', 'inception-v4'], 
    type=str, help='pretrained loss network (default %(default)s)')

  options = parser.parse_args()
  model_identifier = options.network

  loss_network_entity = loss_network(model_identifier)
  model_fn = loss_network_entity.extract_features
  with tf.Graph().as_default():
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    image_string = urllib.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = preprocess_image(image, image_size)
    processed_images  = tf.expand_dims(processed_image, 0)
    tensor_dict = model_fn(processed_images)

    init_fn = loss_network_entity.init_fn
    
    with tf.Session() as sess:
        init_fn(sess)
        tensor_values = sess.run(tensor_dict.values())
    tensor_val_dict = OrderedDict(zip(tensor_dict.keys(), tensor_values))

    print 'Printing {} layers'.format(model_identifier)
    print 'Input image size, {}'.format((1, image_size, image_size, 3))
    for key, value in tensor_val_dict.items():
        print key + ', ' + str(value.shape)

if __name__ == '__main__':
  main()