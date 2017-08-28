import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d
from collections import OrderedDict
import numpy as np, os
from utils import *

class ResizeMethod(object):
  BILINEAR = 0
  NEAREST_NEIGHBOR = 1
  BICUBIC = 2
  AREA = 3

class stylize_network:
  def __init__(self, num_styles, 
               filter_sizes=[9, 3, 3], 
               out_filters=[32, 64, 128], 
               style_filter_size=3,
               conv_separable=False,
               nonlinearity='relu'):
    self._num_styles = num_styles
    assert len(filter_sizes)==3
    assert len(out_filters)==3
    self._filter_sizes = filter_sizes
    self._out_filters = out_filters
    self._style_filter_size = style_filter_size
    self._num_style_filters = out_filters[-1]
    self._MEAN_PIXEL = tf.expand_dims(tf.constant([ 127.5,  127.5,  127.5]), axis=0)
    self._MEAN_PIXEL = tf.expand_dims(tf.expand_dims(self._MEAN_PIXEL, axis=1), axis=2)
    self._conv_separable_bool = conv_separable
    if(self._conv_separable_bool):
      self._conv_op = self._conv_separable
    else:
      self._conv_op = self._conv_basic
    if(nonlinearity=='relu'):
      self._nonlinearity = tf.nn.relu
    elif(nonlinearity=='selu'):
      self._nonlinearity = self.selu
    self._preprocess = self._preprocess_std
    self._postprocess = self._postprocess_std

  def _preprocess_std(self, image):
    return image - self._MEAN_PIXEL

  def _postprocess_std(self, x):
    x = tf.tanh(x)
    images = (x + 1)*127.5
    return images

  def _selu(self, x):
    """selu, self normalizing activation function"""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return tf.identity(scale * tf.where(tf.less(x, 0.0), alpha * tf.nn.elu(x), x), name='selu')

  def _pad_batch(self, x, filter_size):
    pad_left = (filter_size-1)/2
    pad_right = filter_size - 1 - pad_left
    paddings = [[0, 0], [pad_left, pad_right], [pad_left, pad_right], [0, 0]]
    x = tf.pad(x, paddings, "REFLECT")
    return x

  def _instance_norm(self, x, name):
    out_filters = x.get_shape().as_list()[-1]
    instance_mean, instance_var = tf.nn.moments(x, [1,2], keep_dims=True)
    epsilon = 1e-3
    x = (x - instance_mean) / tf.sqrt(instance_var + epsilon)
    scale = tf.get_variable(name=name+'_scale', 
                            shape=[out_filters], 
                            initializer=tf.constant_initializer(1.0))    
    shift = tf.get_variable(name=name+'_shift', 
                            shape=[out_filters], 
                            initializer=tf.constant_initializer(0.01))

    x = x*scale + shift
    return x

  def _cond_instancenorm(self, x, style_ids, name):
    out_filters = x.get_shape().as_list()[-1]
    instance_mean, instance_var = tf.nn.moments(x, [1,2], keep_dims=True)
    epsilon = 1e-3
    x = (x - instance_mean) / tf.sqrt(instance_var + epsilon)
    scale_embeddings = tf.get_variable(name=name+'_scale_embeddings', 
                                       shape=[self._num_styles, out_filters], 
                                       initializer=tf.constant_initializer(1.0))    
    shift_embeddings = tf.get_variable(name=name+'_shift_embeddings', 
                                       shape=[self._num_styles, out_filters], 
                                       initializer=tf.constant_initializer(0.01))

    scale = tf.nn.embedding_lookup(scale_embeddings, style_ids)
    scale = tf.expand_dims(tf.expand_dims(scale, axis=1), axis=2)
    shift = tf.nn.embedding_lookup(shift_embeddings, style_ids)
    shift = tf.expand_dims(tf.expand_dims(shift, axis=1), axis=2)
    x = x*scale + shift
    return x

  def _normalize(self, x, style_ids=None, name='normalize'):
    if style_ids is not None:
      x = self._cond_instancenorm(x, style_ids, name+'_cond_instancenorm')
    else:
      x = self._instance_norm(x, name+'_instance_norm')
    return x

  def _lookup_filter(self, filterbank, style_ids):
    return map(lambda x: tf.nn.embedding_lookup(x, style_ids), filterbank)

  def _conv_basic(self, x, filters, stride=1, 
                  normalize=True, style_ids=None,
                  name='conv_basic'):
    x = tf.nn.conv2d(input=x,
                     filter=filters[0],
                     strides=[1, stride, stride, 1],
                     padding='VALID')
    if(normalize):
      x = self._normalize(x, style_ids=style_ids, 
                         name=name)
    return x

  def _conv_separable(self, x, filters, stride=1, 
                     normalize=True, style_ids=None,
                     name='conv_separable'):
    x = tf.nn.separable_conv2d(input=x,
                               depthwise_filter=filters[0],
                               pointwise_filter=filters[1],
                               strides=[1, stride, stride, 1],
                               padding='VALID')
    if(normalize):
      x = self._normalize(x, style_ids=style_ids, 
                          name=name)
    return x

  def _conv_layer(self, x, filter_size, out_filters, stride=1, 
                  act=True, style_ids=None, normalize=True,
                  name='conv_layer'):
    x = self._pad_batch(x, filter_size)

    in_filters = x.get_shape().as_list()[-1]
    conv_filter = self._find_filter(name, filter_size, in_filters, out_filters)
    x = self._conv_op(x, conv_filter, stride, normalize=normalize, 
                     style_ids=style_ids, name=name)

    if(act):
      x = self._nonlinearity(x)
    return x

  def _residual_unit(self, x_i, style_ids):
    x = self._conv_layer(x_i, self._style_filter_size, 
                        self._num_style_filters, stride=1, 
                        style_ids=style_ids, 
                        name='conv1')
    x = self._conv_layer(x, self._style_filter_size, 
                        self._num_style_filters, stride=1, 
                        act=False, style_ids=style_ids,
                        name='conv2')
    x = self._nonlinearity(x + x_i)
    return x

  def _encoder(self, x, style_ids=None, cond_instancenorm=False):
    filter_sizes = self._filter_sizes
    num_filters = self._out_filters
    acts = [True, True, True]
    normalize = [True, True,True]
    strides = [1, 2, 2]

    x = self._preprocess(x)
    for i, tup in enumerate(zip(filter_sizes, num_filters, acts, strides, normalize)):
      with tf.variable_scope('encode_' + str(i+1)) as scope:
        x = self._conv_layer(x, tup[0], tup[1], act=tup[2],
                            stride=tup[3], style_ids=style_ids,
                            normalize=tup[4])
    return x

  def _residual_block(self, x, style_ids):
    with tf.variable_scope('residual'):
      for i in range(4):
        with tf.variable_scope('res_'+str(i+1)):
          x = self._residual_unit(x, style_ids)
    return x

  def _decoder(self, x, normalize=True, style_ids=None, resize_factor=0):
    filter_sizes = list(reversed(self._filter_sizes))
    filtersize1, filtersize2 = filter_sizes[-3], [filter_sizes[-2]]

    num_filters = list(reversed(self._out_filters))[1:] + [3]
    numfilters1, numfilters2, numfilters3 = num_filters[-3], [num_filters[-2]], num_filters[-1]

    filter_sizes = filter_sizes[:-2] + [filtersize1]*resize_factor + filtersize2 + [3]
    num_filters = num_filters[:-2] + [numfilters1]*resize_factor + numfilters2 + [numfilters3]
    acts = [True] + [True]*resize_factor + [True] + [False]
    upsample = [True] + [True]*resize_factor + [True] + [False]

    for i, tup in enumerate(zip(filter_sizes, num_filters, acts, upsample)):
      with tf.variable_scope('decode_' + str(i+1)) as scope:
        x = self._conv_layer(x, tup[0], tup[1], act=tup[2], 
                            normalize=normalize, style_ids=style_ids)
        if(tup[3]):
          shp = tf.shape(x)
          x = tf.image.resize_nearest_neighbor(x, (2*shp[1], 2*shp[2]))
    x = self._postprocess(x)
    return x

  def _find_filter(self, name, filter_size, in_filters, out_filters):
    if(self._conv_separable_bool):
      depthwise_conv_filter = tf.get_variable(name=name+'_depthwise_weights',
                                              shape=[filter_size, filter_size, in_filters, 1],
                                              initializer=tf.random_normal_initializer(stddev=0.01))
      pointwise_conv_filter = tf.get_variable(name=name+'_pointwise_weights',
                                              shape=[1, 1, in_filters, out_filters],
                                              initializer=tf.random_normal_initializer(stddev=0.01))
      return (depthwise_conv_filter, pointwise_conv_filter)
    else:
      conv_filter = tf.get_variable(name=name+'_conv_weights',
                                    shape=[filter_size, filter_size, in_filters, out_filters],
                                    initializer=tf.random_normal_initializer(stddev=0.1))
      return (conv_filter,)

  def _resize_images(self, x, size_factor=2, 
                    method=ResizeMethod.NEAREST_NEIGHBOR):
    shp = tf.shape(x)
    new_size = (tf.to_int32(tf.constant(size_factor)*tf.to_float(shp[1])), 
                tf.to_int32(tf.constant(size_factor)*tf.to_float(shp[2])))
    x = tf.image.resize_images(x, new_size, method=method)
    return x

  def stylize(self, x, style_ids, decoder_norm=False, resize_factor=0):
    size_factor = 1/float(2**resize_factor)
    x = self._resize_images(x, size_factor=size_factor, method=ResizeMethod.BILINEAR)
    x = self._encoder(x, style_ids=style_ids)
    x = self._residual_block(x, style_ids)
    x = self._decoder(x, normalize=decoder_norm, style_ids=style_ids, resize_factor=resize_factor)
    imgs = tf.identity(x, name='combined_images')
    return imgs
