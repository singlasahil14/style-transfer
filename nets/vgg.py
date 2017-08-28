from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def vgg_arg_scope(padding='SAME'):
  with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding=padding) as arg_sc:
      return arg_sc

def preprocess(images):
  images = images - tf.constant([ _R_MEAN ,  _G_MEAN,  _B_MEAN])
  return images

def repeat(inputs, repetitions, layer, *args, **kwargs):
  scope = kwargs.pop('scope', 'conv')
  end_points = kwargs.pop('end_points', OrderedDict())
  with tf.variable_scope(scope, 'Repeat', [inputs]):
    inputs = tf.convert_to_tensor(inputs)
    outputs = inputs
    for i in range(repetitions):
      scope_name = scope + '_' + str(i+1)
      kwargs['scope'] = scope_name
      outputs = layer(outputs, *args, **kwargs)
      end_points[scope_name] = outputs
    return outputs, end_points

def vgg_16(inputs, scope='vgg_16', reuse=False, pool_fn=slim.max_pool2d):
  with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    # Collect outputs for conv2d and pool_fn.
    with slim.arg_scope([slim.conv2d, pool_fn]):
      net, end_points = repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = pool_fn(net, [2, 2], scope='pool1')
      end_points['pool1'] = net
      net, end_points = repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool2')
      end_points['pool2'] = net
      net, end_points = repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool3')
      end_points['pool3'] = net
      net, end_points = repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool4')
      end_points['pool4'] = net
      net, end_points = repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool5')
      end_points['pool5'] = net
      return end_points

def vgg_19(inputs, scope='vgg_19', reuse=False, pool_fn=slim.max_pool2d):
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    # Collect outputs for conv2d and pool_fn.
    with slim.arg_scope([slim.conv2d, pool_fn]):
      net, end_points = repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = pool_fn(net, [2, 2], scope='pool1')
      end_points['pool1'] = net
      net, end_points = repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool2')
      end_points['pool2'] = net
      net, end_points = repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool3')
      end_points['pool3'] = net
      net, end_points = repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool4')
      end_points['pool4'] = net
      net, end_points = repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5', end_points=end_points)
      net = pool_fn(net, [2, 2], scope='pool5')
      end_points['pool5'] = net
      return end_points