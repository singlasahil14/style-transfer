# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import inception
from nets import vgg

slim = tf.contrib.slim

networks_map = {'vgg-16': vgg.vgg_16,
                'vgg-19': vgg.vgg_19,
                'inception-v1': inception.inception_v1,
                'inception-v2': inception.inception_v2,
                'inception-v3': inception.inception_v3,
                'inception-v4': inception.inception_v4
               }

arg_scopes_map = {'vgg-16': vgg.vgg_arg_scope,
                  'vgg-19': vgg.vgg_arg_scope,
                  'inception-v1': inception.inception_arg_scope,
                  'inception-v2': inception.inception_arg_scope,
                  'inception-v3': inception.inception_arg_scope,
                  'inception-v4': inception.inception_arg_scope
                 }

preprocessing_fn_map = {'vgg': vgg.preprocess,
                        'inception': inception.preprocess,
                       }

scale_mul_map = {'vgg': 1.,
                 'inception': 127.5,
                }

pool_fn_map = {'max': slim.max_pool2d,
               'avg': slim.avg_pool2d,
              }


def get_preprocessing_fn(name):
  arch = name.split('-')[0]
  preprocessing_fn = preprocessing_fn_map[arch]
  return preprocessing_fn

def get_scalar_mul(name):
  arch = name.split('-')[0]
  scale_mul = scale_mul_map[arch]
  return scale_mul

def get_network_fn(name):
  """Returns a network_fn such as `end_points = network_fn(images)`.

  Args:
    name: The name of the network.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  preprocessing_fn = get_preprocessing_fn(name)
  @functools.wraps(func)
  def network_fn(images, reuse=None, pool_layer='max', padding='SAME'):
    assert padding in ['SAME', 'VALID']
    pool_fn = pool_fn_map[pool_layer]
    images = preprocessing_fn(images)
    arg_scope = arg_scopes_map[name](padding=padding)
    with slim.arg_scope(arg_scope):
      return func(images, reuse=reuse, pool_fn=pool_fn)
  return network_fn
