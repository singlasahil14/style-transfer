from loss_network_factory import *
import time
import bcolz
from itertools import cycle
from utils import *
from collections import defaultdict, OrderedDict
from style_network_factory import *

import pandas as pd
import tensorflow as tf, numpy as np, os
import scipy
from collections import defaultdict
from tensorflow.contrib.opt.python.training import external_optimizer

STYLE_WEIGHT_MULTIPLIER = 1e-4

class LossMinimizer:
  def __init__(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)
    self._input_images = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='input_images')
    self._non_trainable_vars = []

  def setup_feature_fn(self, model_identifier='vgg-16', pool_layer='avg', padding='SAME', content_layers=[], style_layers=[]):
    self._loss_network_attrs = {'loss_network': model_identifier, 'pool_layer': pool_layer, 'padding': padding}
    loss_network_entity = loss_network(model_identifier)
    _features_fn = loss_network_entity.extract_features
    self._features_dict = _features_fn(self._input_images, pool_layer=pool_layer, padding=padding)
    self._features_fn = lambda x: _features_fn(x, reuse=True, pool_layer=pool_layer, padding=padding)
    loss_network_entity.init_fn(self._sess)
    self._scalar_mul = loss_network_entity.scalar_mul
    self._non_trainable_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=loss_network_entity.model_scope)
    self.all_layers = self._features_dict.keys()
    self.content_layers = loss_network_entity.content_layers
    if(len(content_layers)>0):
      self.content_layers = [x for x in self.all_layers if x in content_layers]
      assert len(self.content_layers)==len(content_layers), "specified content_layers not present in loss network"

    self.style_layers = loss_network_entity.style_layers
    if(len(style_layers)>0):
      self.style_layers = [x for x in self.all_layers if x in style_layers]
      assert len(self.style_layers)==len(style_layers), "specified style_layers not present in loss network"

  def compute_style_features(self, style_path, style_image_size=None):
    self.style_names, self.style_images = load_images(style_path, style_image_size)
    self._target_style = self._find_target(self.style_images, type='style')
    self.num_styles = len(self.style_names)

  def _setup_result_path(self, result_path):
    self.result_path = result_path
    os.makedirs(self.result_path)
    self.images_path = os.path.join(self.result_path, 'images')
    os.makedirs(self.images_path)

  def _gram_matrix(self, features_tensor):
    batch_size, height, width, num_channels = tensor_shape(features_tensor)
    feature_map_size = height * width 

    features = tf.reshape(features_tensor, (batch_size, height*width, num_channels))
    features_T = tf.transpose(features, perm=[0,2,1])

    gram = tf.matmul(features_T, features)/(tf.to_float(feature_map_size))
    return gram

  def _extract_style(self, features_dict):
    style_features = OrderedDict([(feature_name, self._gram_matrix(features_dict[feature_name])) for feature_name in features_dict.keys()])
    return style_features

  def _find_target(self, images, type='style'):
    tensor_values = []
    if(type=='style'):
      tensor_dict = self._extract_style(self._features_dict)
    elif(type=='content'):
      tensor_dict = self._features_dict
    else:
      raise ValueError('Unknown type. Should be one of style/content.')
    for image in images:
      feed_image = np.expand_dims(image, axis=0)
      single_tensor_values = self._sess.run(tensor_dict.values(), feed_dict={self._input_images: feed_image})
      tensor_values.append(single_tensor_values)

    tensor_values = zip(*tensor_values)
    tensor_values = map(np.concatenate, tensor_values)
    tensor_dict = dict(zip(tensor_dict.keys(), map(tf.constant, tensor_values)))
    return tensor_dict

  def _setup_stylewise_loss_weights(self, default_style_weight, default_tv_weight, 
                                    stylewise_style_weights_dict={}, stylewise_tv_weights_dict={}):
    style_weights_np = np.asarray([default_style_weight]*self.num_styles)
    tv_weights_np = np.asarray([default_tv_weight]*self.num_styles)
    for i,style_name in enumerate(self.style_names):
      if(style_name in stylewise_style_weights_dict):
         style_weights_np[i] = stylewise_style_weights_dict[style_name]
      if(style_name in stylewise_tv_weights_dict):
         tv_weights_np[i] = stylewise_tv_weights_dict[style_name]
    self.style_weights = tf.constant(style_weights_np, dtype=tf.float32)
    self.tv_weights = tf.constant(tv_weights_np, dtype=tf.float32)

  def _setup_layerwise_loss_weights(self, layerwise_content_weights_dict={}, layerwise_style_weights_dict={}, 
      default_layerwise_content_weight=1., default_layerwise_style_weight=1.):
    assert set(layerwise_content_weights_dict.keys()).issubset(self.content_layers)
    self.layerwise_content_weights_dict = OrderedDict()
    for key in self.content_layers:
      if key in layerwise_content_weights_dict:
        self.layerwise_content_weights_dict[key] = layerwise_content_weights_dict[key]
      else:
        self.layerwise_content_weights_dict[key] = default_layerwise_content_weight

    assert set(layerwise_style_weights_dict.keys()).issubset(self.style_layers)
    self.layerwise_style_weights_dict = OrderedDict()
    for key in self.style_layers:
      if key in layerwise_style_weights_dict:
        self.layerwise_style_weights_dict[key] = layerwise_style_weights_dict[key]
      else:
        self.layerwise_style_weights_dict[key] = default_layerwise_style_weight

  def _find_content_style_loss(self, pred_batch, target_content_batch, target_style_batch):
    self._layerwise_content_losses_dict = OrderedDict()
    self._layerwise_style_losses_dict = OrderedDict()
    style_weights_batch = tf.nn.embedding_lookup(self.style_weights, self._train_style_ids)

    for key in pred_batch.keys():
      batch_size, height, width, num_channels = tensor_shape(pred_batch[key])
      if key in target_content_batch:
        layer_content_loss = self.content_weight*tf.nn.l2_loss(pred_batch[key]-target_content_batch[key])
        self._layerwise_content_losses_dict[key] = layer_content_loss/tf.to_float(batch_size*height*width*num_channels)

      pred_style_value = self._gram_matrix(pred_batch[key])
      style_losses = tf.map_fn(lambda x: x[2]*tf.nn.l2_loss(x[0]-x[1]), 
                      (pred_style_value, target_style_batch[key], style_weights_batch), dtype=tf.float32)
      layer_style_loss = tf.reduce_sum(style_losses)/tf.to_float(batch_size*num_channels*num_channels)
      self._layerwise_style_losses_dict[key] = STYLE_WEIGHT_MULTIPLIER*layer_style_loss

    self._wtd_layerwise_content_losses_dict = OrderedDict()
    content_loss = 0
    for key in self.content_layers:
      layer_content_weight = self.layerwise_content_weights_dict[key]
      weighted_layer_content_loss = layer_content_weight*self._layerwise_content_losses_dict[key]
      self._wtd_layerwise_content_losses_dict[key] = weighted_layer_content_loss
      content_loss += weighted_layer_content_loss

    self._wtd_layerwise_style_losses_dict = OrderedDict()
    style_loss = 0
    for key in self.style_layers:
      layer_style_weight = self.layerwise_style_weights_dict[key]
      weighted_layer_style_loss = layer_style_weight*self._layerwise_style_losses_dict[key]
      self._wtd_layerwise_style_losses_dict[key] = weighted_layer_style_loss
      style_loss += weighted_layer_style_loss
    return content_loss, style_loss

  def _find_tv_loss(self, preds):
    tv_weights_batch = tf.nn.embedding_lookup(self.tv_weights, self._train_style_ids)    

    batch_size, height, width, num_channels = tensor_shape(preds)
    tv_y_size = tf.to_float((height-1)*width*num_channels)
    tv_x_size = tf.to_float(height*(width-1)*num_channels)
    y_tv_loss = tf.multiply(tv_weights_batch, tf.map_fn(tf.nn.l2_loss, preds[:,1:,:,:] - preds[:,:height-1,:,:]))
    x_tv_loss = tf.multiply(tv_weights_batch, tf.map_fn(tf.nn.l2_loss, preds[:,:,1:,:] - preds[:,:,:width-1,:]))
    tv_loss = tf.reduce_sum(x_tv_loss/tv_x_size + y_tv_loss/tv_y_size)
    tv_loss = tv_loss/tf.to_float(batch_size)
    return tv_loss

  def setup_loss(self, content_weight, default_style_weight, default_tv_weight, stylewise_style_weights_dict={}, stylewise_tv_weights_dict={}, 
      layerwise_content_weights_dict={}, layerwise_style_weights_dict={}, default_layerwise_content_weight=1., default_layerwise_style_weight=1.):
    self.train_metrics = OrderedDefaultListDict()

    self.content_weight = content_weight
    self._setup_stylewise_loss_weights(default_style_weight, default_tv_weight, stylewise_style_weights_dict, stylewise_tv_weights_dict)
    self._setup_layerwise_loss_weights(layerwise_content_weights_dict, layerwise_style_weights_dict, 
      default_layerwise_content_weight=default_layerwise_content_weight, default_layerwise_style_weight=default_layerwise_style_weight)

    pred_batch = self._features_fn(self._stylized_train_images)
    target_style_batch = dict([(k, tf.nn.embedding_lookup(v, self._train_style_ids)) for k, v in self._target_style.items()])

    self._content_loss, self._style_loss = self._find_content_style_loss(pred_batch, self._target_content_batch, target_style_batch)
    self._tv_loss = self._find_tv_loss(self._stylized_train_images)

    self._total_loss = self._content_loss + self._style_loss + self._tv_loss

  def _log_losses(self, losses, layerwise_content_losses_dict=OrderedDict(), layerwise_style_losses_dict=OrderedDict()):
    total_loss, content_loss, style_loss, tv_loss = losses
    self.train_metrics['total_loss'].append(total_loss)
    self.train_metrics['content_loss'].append(content_loss)
    self.train_metrics['style_loss'].append(style_loss)
    self.train_metrics['tv_loss'].append(tv_loss)

    for key, value in layerwise_content_losses_dict.items():
      self.train_metrics['content_'+key].append(value)

    for key, value in layerwise_style_losses_dict.items():
      self.train_metrics['style_'+key].append(value)

  def _print_losses_dict(self, losses_dict):
    print_str = ''
    for key, value in losses_dict.items():
      print_str = print_str + '%s: %s, ' % (key, value)
    print(print_str[:-2])

  def _print_content_and_style_layerwise_losses(self, content_val_dict, style_val_dict, prefix_str='layerwise'):
    if(len(content_val_dict)>0):
      print(prefix_str + ' content losses')
      self._print_losses_dict(content_val_dict)
    if(len(style_val_dict)>0):
      print(prefix_str + ' style losses')
      self._print_losses_dict(style_val_dict)
    print('')

  def _print_values(self, losses):
    pd_train_metrics = pd.DataFrame(self.train_metrics)
    pd_train_metrics.to_csv(os.path.join(self.result_path, 'train_metrics.csv'))
    print('total: %s, content:%s, style: %s, tv: %s' % losses)

class SlowLossMinimizer(LossMinimizer):
  def __init__(self):
    LossMinimizer.__init__(self)

  def compute_content_style_features(self, content_path, style_path, content_image_size=None, style_image_size=None):
    self.content_image_size = content_image_size
    self.content_image = load_image(content_path, image_size=content_image_size)
    self._target_content_batch = self._find_target([self.content_image], type='content')
    self.compute_style_features(style_path, style_image_size=style_image_size)

  def calc_all_layer_losses(self, print_losses=False):
    self._sess.run(self._stylized_image.initializer)
    content_vals = self._sess.run(self._layerwise_content_losses_dict.values())
    content_val_dict = OrderedDict(zip(self._layerwise_content_losses_dict.keys(), content_vals))
    style_vals = self._sess.run(self._layerwise_style_losses_dict.values())
    style_val_dict = OrderedDict(zip(self._layerwise_style_losses_dict.keys(), style_vals))
    if print_losses:
      print('Printing content and style losses for all layers')
      self._print_content_and_style_layerwise_losses(content_val_dict, style_val_dict)
    return content_val_dict, style_val_dict

  def setup_compute_graph(self, initial_identifier='content'):
    if(initial_identifier=='noise'):
      init_image = np.random.uniform(-0.5, 0.5, np.shape(self.content_image)) + 127.5
    elif(initial_identifier=='content'):
      init_image = self.content_image
    else:
      init_image = load_image(initial_identifier, image_size=self.content_image_size)
      assert init_image.shape == self.content_image.shape
    self._stylized_image = tf.Variable(init_image, dtype=tf.float32)
    self._stylized_train_images = tf.expand_dims(self._stylized_image, axis=0)
    self._train_style_ids = tf.constant([0])

  def run_optimization(self, total_iterations=1000, checkpoint_iterations=100, result_path='result'):
    self._setup_result_path(result_path)
    self.iters = 0
    self.checkpoint_iterations = checkpoint_iterations
    optimizer = external_optimizer.ScipyOptimizerInterface(self._total_loss, var_list=[self._stylized_image], options={'maxiter': total_iterations})
    self._sess.run(self._stylized_image.initializer)
    print('Started optimization\n')
    self._start_time = time.time()
    optimizer.minimize(session=self._sess, loss_callback=self._output_results, fetches=[self._total_loss, self._content_loss, self._style_loss, 
      self._tv_loss, self._wtd_layerwise_content_losses_dict.values(), self._wtd_layerwise_style_losses_dict.values(), self._stylized_image])

  def _save_image(self, stylized_image):
    final_path = os.path.join(self.images_path, str(self.iters)+'.jpg' )
    save_image(final_path, stylized_image)

  def _output_results(self, total_loss, content_loss, style_loss, tv_loss, wtd_layerwise_content_losses, wtd_layerwise_style_losses, 
      stylized_image):
    wtd_layerwise_content_losses_dict = OrderedDict(zip(self._wtd_layerwise_content_losses_dict.keys(), wtd_layerwise_content_losses))
    wtd_layerwise_style_losses_dict = OrderedDict(zip(self._wtd_layerwise_style_losses_dict.keys(), wtd_layerwise_style_losses))
    self._log_losses((total_loss, content_loss, style_loss, tv_loss), wtd_layerwise_content_losses_dict, wtd_layerwise_style_losses_dict)
    self.iters = self.iters + 1    
    if((self.iters-1)%self.checkpoint_iterations==0):
      avg_time_per_iteration = (time.time() - self._start_time)/float(self.iters)
      print('iteration: %d, average time/iteration: %f' % (self.iters, avg_time_per_iteration))
      self._print_values((total_loss, content_loss, style_loss, tv_loss))
      self._print_content_and_style_layerwise_losses(wtd_layerwise_content_losses_dict, wtd_layerwise_style_losses_dict)
      self._save_image(stylized_image)

class FastLossMinimizer(LossMinimizer):
  def __init__(self):
    LossMinimizer.__init__(self)
    self._is_gpu_present = len(get_available_gpus()) > 0

  def _setup_models_and_test_results_path(self, test_path):
    self.models_path = os.path.join(self.result_path, 'models')
    os.makedirs(self.models_path)
    if(test_path is None):
      self.test_image_names, self.test_images = [], []
      return
    self.test_image_names, self.test_images = load_images(test_path)
    for style_name in self.style_names:
      create_subfolder(self.images_path, style_name)
      for test_image_name in self.test_image_names:
        style_result_path = os.path.join(self.images_path, style_name)
        create_subfolder(style_result_path, test_image_name)

  def compute_content_features(self, train_path, result_path, sq_size=256, batch_size=16, subset_size=None):
    data_reader = data_pipeline(train_path, sq_size=sq_size, batch_size=batch_size, subset_size=subset_size)
    iterator = data_reader.iterator
    feed_images = iterator.get_next()
    features_dict = self._features_fn(feed_images)
    feature_keys = ['image'] + list(self.content_layers)
    content_tensors = [features_dict[key] for key in self.content_layers]

    os.makedirs(result_path)
    self._sess.run(iterator.initializer)
    i = 0
    while True:
      try:
        feature_values = self._sess.run([feed_images] + content_tensors)
        if(i==0):
          bcolz_arrs_dict = open_bcolz_arrays(result_path, feature_keys, feature_values, mode='w', attr_dict=self._loss_network_attrs)
        else:
          for key, value in zip(feature_keys, feature_values):
            bcolz_arr = bcolz_arrs_dict[key]
            bcolz_arr.append(value)
            bcolz_arr.flush()
        i = i + 1
      except tf.errors.OutOfRangeError:
        break

  def setup_data_pipeline(self, train_path, image_size=256, batch_size=16, subset_size=None):
    data_reader = data_pipeline(train_path, sq_size=image_size, batch_size=batch_size, subset_size=subset_size)
    self.iterator = data_reader.iterator
    self._train_images = self.iterator.get_next()
    self._target_content_batch = self._features_fn(self._train_images)
    self._train_style_ids = tf.random_uniform([tf.shape(self._train_images)[0]], maxval=self.num_styles, dtype=tf.int32)

  def setup_features_pipeline(self, train_path, batch_size=16, subset_size=None):
    feature_keys = ['image'] + self.content_layers
    features_reader = features_pipeline(train_path, feature_keys, batch_size=batch_size, attr_dict=self._loss_network_attrs)
    self.iterator = features_reader.iterator
    feature_values = self.iterator.get_next()
    self._train_images = feature_values[0]
    content_values = feature_values[1:]
    self._target_content_batch = dict(zip(self.content_layers, content_values))
    self._train_style_ids = tf.random_uniform([tf.shape(self._train_images)[0]], maxval=self.num_styles, dtype=tf.int32)

  def setup_compute_graph(self, conv_separable=False, nonlinearity='relu', decoder_norm=True, resize_factor=0, variable_scope_name='transform'):
    self.variable_scope_name = variable_scope_name
    stylize_network_entity = stylize_network(self.num_styles, conv_separable=conv_separable, nonlinearity=nonlinearity)
    self._stylize_fn = lambda x,y: stylize_network_entity.stylize(x, y, decoder_norm=decoder_norm, resize_factor=resize_factor)
    self._style_ids = tf.placeholder(shape=(None,), dtype=tf.int32, name='style_ids')
    with tf.variable_scope(self.variable_scope_name):
      self._eval_images = self._stylize_fn(self._input_images, self._style_ids)
    with tf.variable_scope(self.variable_scope_name, reuse=True):
      self._stylized_train_images = self._stylize_fn(self._train_images, self._train_style_ids)

  def _checkpoint_model(self):
    saver = tf.train.Saver()
    model_name = 'epoch_{},iter_{}'.format(self.epoch_i, self.iters)
    model_filepath = os.path.join(self.models_path, model_name)
    saver.save(self._sess, model_filepath, write_meta_graph=False, write_state=False)

  def _checkpoint_test_images(self):
    img_name =  'epoch_' + str(self.epoch_i) + ', iter_' + str(self.iters) + '.jpg'
    for j, style_name in enumerate(self.style_names):
      style_result_path = os.path.join(self.images_path, style_name)
      for test_image_name, test_image in zip(self.test_image_names, self.test_images):
        [new_image] = self._sess.run([self._eval_images], feed_dict={self._style_ids:[j], self._input_images:[test_image]})
        image_to_save = np.squeeze(new_image, axis=0)
        self._save_test_image(style_result_path, test_image_name, img_name, image_to_save)

  def _checkpoint(self, style_losses):
    self._print_values(style_losses)
    self._checkpoint_model()
    self._checkpoint_test_images()

  def _print_style_names(self):
    print('Print style ids and corresponding style names')
    for style_id,style_name in enumerate(self.style_names):
      print('style id: %d, style name: %s' % (style_id, style_name))
    print('')

  def _print_inference_times(self):
    print('Print inference times for test images')
    with tf.device('/cpu:0'):
      with tf.variable_scope(self.variable_scope_name, reuse=True):
        _cpu_eval_images = self._stylize_fn(self._input_images, self._style_ids)
      for test_image in self.test_images:
        height, width, depth = test_image.shape
        start_time = time.time()
        self._sess.run(_cpu_eval_images, feed_dict={self._style_ids: [0], self._input_images: [test_image]})
        print('CPU inference time for %dx%dx%d image: %f' % (height, width, depth, time.time()-start_time))
        if self._is_gpu_present:
          start_time = time.time()
          self._sess.run(self._eval_images, feed_dict={self._style_ids: [0], self._input_images: [test_image]})
          print('GPU inference time for %dx%dx%d image: %f' % (height, width, depth, time.time()-start_time))
    print('')

  def setup_train_step(self, learning_rate=1e-3):
    trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.variable_scope_name)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    complete_grads, complete_vars = zip(*optimizer.compute_gradients(self._total_loss, var_list=trainable_vars))
    self._complete_grads_norm = tf.global_norm(complete_grads)
    self._train_op = optimizer.apply_gradients(zip(complete_grads, complete_vars))

  def run_optimization(self, result_path, total_epochs=4, checkpoint_iterations=100, test_path=None):
    self._setup_result_path(result_path)
    self._setup_models_and_test_results_path(test_path)
    all_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    variables_to_initialize = list(set(all_vars_list) - set(self._non_trainable_vars))
    self._sess.run(tf.variables_initializer(variables_to_initialize))
    self._print_style_names()
    self._print_inference_times()
    print('Started training')
    for self.epoch_i in range(total_epochs):
      self._sess.run(self.iterator.initializer)
      self.iters = 0
      start_time = time.time()
      while True:
        try:
          sess_run_values = self._sess.run([self._train_op, self._total_loss, self._content_loss, self._style_loss, self._tv_loss, self._complete_grads_norm])
          grad_norm_value = sess_run_values[-1]
    #      print grad_norm_value
          style_values = tuple(sess_run_values[1:5])
          self._log_losses(style_values)
          self.iters = self.iters + 1
          if((self.iters-1)%checkpoint_iterations==0):
            time_per_iteration = (time.time()-start_time)/float(self.iters)
            print('epoch %d, iteration: %d: average time/iteration: %f' % (self.epoch_i, self.iters, time_per_iteration))
            self._checkpoint(style_values)
        except tf.errors.OutOfRangeError:
          break
      print("Time per epoch: " + str(time.time()-start_time))
    print('Done training')

  def _save_test_image(self, dir_path, subdir_name, img_name, img):
    out_path = os.path.join(dir_path, subdir_name, img_name)
    img = np.clip(img, 0, 255).astype(np.uint8)
    save_image(out_path, img)
