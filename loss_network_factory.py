import tensorflow as tf
import numpy as np, os
from collections import OrderedDict
import scipy.io
from utils import tensor_shape, download_model
from nets import nets_factory

slim = tf.contrib.slim

allowed_model_architectures = {'vgg-16', 'vgg-19', 'inception-v1', 'inception-v2', 'inception-v3', 'inception-v3-openimages', 
  'inception-v4'}

model_url_dict = {
  'vgg-16': 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz', 
  'vgg-19': 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
  'inception-v1': 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
  'inception-v2': 'http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz',
  'inception-v3': 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
  'inception-v3-openimages': 'https://storage.googleapis.com/openimages/2016_08/model_2016_08.tar.gz',
  'inception-v4': 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz'}
assert (set(model_url_dict.keys())==allowed_model_architectures)

def _find_model_alias(model_identifier):
  model_name_split = model_identifier.split('-')
  model_alias = '-'.join(model_name_split[:2])
  return model_alias

def _find_model_name(model_identifier):
  model_identifier_split = model_identifier.split('-')
  model_name_split = model_identifier_split + ['imagenet']
  model_name = '-'.join(model_name_split[:3])
  return model_name

def _find_scope_name(model_alias):
  model_alias_split = model_alias.split('-')
  model_arch = model_alias_split[0]
  if(model_arch=='vgg'):
    scope_name = '_'.join(model_alias_split)
  elif(model_arch=='inception'):
    str_list = [x.title() for x in model_alias_split]
    scope_name = ''.join(str_list)
  else:
    raise ValueError('Unknown model architecture')
  return scope_name

_content_layers_dict = {
  'vgg-16': ('conv2_2',), 
  'vgg-19': ('conv2_2',),
  'inception-v1': ('Conv2d_2c_3x3',),
  'inception-v2': ('Conv2d_2c_3x3',),
  'inception-v3': ('Conv2d_4a_3x3',),
  'inception-v4': ('Mixed_3a',),
  }

_style_layers_dict = {
  'vgg-16': ('conv3_1', 'conv4_1', 'conv5_1'), 
  'vgg-19': ('conv2_2', 'conv3_1', 'conv4_1', 'conv5_1'),
  'inception-v1': ('Conv2d_2c_3x3', 'Mixed_3c', 'Mixed_4b', 'Mixed_5b'),
  'inception-v2': ('Conv2d_2c_3x3', 'Mixed_3b', 'Mixed_4a', 'Mixed_5a'),
  'inception-v3': ('Conv2d_4a_3x3', 'Mixed_5b', 'Mixed_6a', 'Mixed_7a'),
  'inception-v4': ('Mixed_4a', 'Mixed_5a', 'Mixed_6a', 'Mixed_7a'),
  }

class loss_network:
  def __init__(self, model_identifier):
    assert model_identifier in allowed_model_architectures
    model_name = _find_model_name(model_identifier)
    model_alias = _find_model_alias(model_identifier)
    self._model_fn = nets_factory.get_network_fn(model_alias)
    self.scalar_mul = nets_factory.get_scalar_mul(model_identifier)

    self.content_layers = _content_layers_dict[model_alias]
    self.style_layers = _style_layers_dict[model_alias]

    model_dir ='data/lossnet'
    model_url = model_url_dict[model_identifier]
    download_model(model_url, model_dir, model_name)

    self._model_path = os.path.join(model_dir, model_name + '.ckpt')
    model_alias = _find_model_alias(model_name)
    self.model_scope = _find_scope_name(model_alias)

  def init_fn(self, sess):
    init = slim.assign_from_checkpoint_fn(self._model_path, slim.get_model_variables(self.model_scope))
    init(sess)

  def extract_features(self, inp, reuse=None, pool_layer='avg', padding='SAME'):
    features_dict = self._model_fn(inp, reuse=reuse, pool_layer=pool_layer, padding=padding)
    features_dict = OrderedDict(zip(features_dict.keys(), map(lambda x: tf.scalar_mul(self.scalar_mul, x), features_dict.values())))
    return features_dict
