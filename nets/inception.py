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
"""Brings all inception models under one namespace."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from nets.inception_v1 import inception_v1
from nets.inception_v1 import inception_v1_base
from nets.inception_v2 import inception_v2
from nets.inception_v2 import inception_v2_base
from nets.inception_v3 import inception_v3
from nets.inception_v3 import inception_v3_base
from nets.inception_v4 import inception_v4
from nets.inception_v4 import inception_v4_base
# pylint: enable=unused-import

import tensorflow as tf
slim = tf.contrib.slim

def preprocess(images):
  images = images/255.
  images = tf.subtract(images, 0.5)
  images = tf.multiply(images, 2.0)
  return images

def inception_arg_scope(use_batch_norm=True, batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001, padding=None):
  """Defines the default arg scope for inception models.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.

  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
    # Decay for the moving averages.
    'decay': batch_norm_decay,
    # epsilon to prevent 0s in variance.
    'epsilon': batch_norm_epsilon,
    # collection containing update_ops.
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}

  with slim.arg_scope([slim.conv2d, slim.fully_connected]):
    with slim.arg_scope(
      [slim.conv2d],
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=normalizer_params) as sc:
      return sc
