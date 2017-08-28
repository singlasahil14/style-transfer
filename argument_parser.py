import os
from argparse import ArgumentParser

from utils import *

loss_network_choices = ['vgg-16', 'vgg-19', 'inception-v1', 'inception-v2', 'inception-v3', 'inception-v3-openimages', 'inception-v4']
DEFAULT_STYLE_WEIGHT = 2e2

pool_layer_choices = ['max', 'avg']
INCEPTION_DEFAULT_POOL_LAYER = 'avg'
OTHER_DEFAULT_POOL_LAYER = 'max'

padding_choices = ['SAME', 'VALID']

TRAIN_IMAGE_SIZE = 256

class _Parser(ArgumentParser):
  def add_content_arguments(self):
    self.add_argument('--content-path', type=str, help='content image path', required=True)
    self.add_argument('--content-image-size', default=None, type=int, help='content image is cropped centrally to this size (default %(default)s)')

  def add_result_arguments(self):
    self.add_argument('--result-path', type=str, help='path to results directory', required=True)

  def add_test_arguments(self):
    self.add_argument('--test-path', type=str, help='path to test image or images (default: %(default)s)', required=True)

  def add_style_arguments(self):
    self.add_argument('--style-path', type=str, help='style image path', required=True)
    self.add_argument('--style-image-size', default=None, type=int, help='size of style image to train with (default %(default)s)')

  def add_loss_network_arguments(self):
    self.add_argument('--loss-network', default='vgg-16', choices=loss_network_choices, type=str, help='pretrained loss network (default %(default)s)')
    self.add_argument('--pool-layer', default='avg', choices=pool_layer_choices, type=str, help='pooling type in loss network (default %(default)s)')
    self.add_argument('--padding', default='SAME', choices=padding_choices, type=str, 
                        help='padding type in loss network (for vgg16/vgg19 only) (default %(default)s)')

  def add_loss_weights_arguments(self):
    self.add_argument('--content-weight', default=8e0, type=float, help='content weight (default %(default)s)')
    self.add_argument('--style-weight', default=None, type=float, help='style weight (default %(default)s)')
    self.add_argument('--tv-weight', default=2e2, type=float, help='total variation loss weight (default %(default)s)')

  def add_layerwise_loss_arguments(self):
    self.add_argument('--content-layers', nargs='+', default=[], type=str, 
                      help='content layers for finding the content loss (default: as specified in loss_network_factory.py)')
    self.add_argument('--layerwise-content-weights', nargs='+', default=[], type=float, 
                      help='respective weights for losses of content layers, 1. if unspecified')
    self.add_argument('--style-layers', nargs='+', dest='style_layers', metavar='STYLE_LAYERS', default=[], type=str, 
                      help='style layers for finding the style loss (default: as specified in loss_network_factory.py)')
    self.add_argument('--layerwise-style-weights', nargs='+', default=[], type=float, 
                      help='respective weights for losses of style layers, 1. if unspecified (all weights will be multiplied by 1e-4 on training)')

  def add_stylewise_loss_arguments(self):
    self.add_argument('--style-names', nargs='+', default=[], type=str, 
                      help='style names for non-default style weights and tv weights (default: %(default)s)')
    self.add_argument('--nondefault-style-weights', nargs='+', default=[], type=float, 
                      help='style weights for style names specified (default: %(default)s)')
    self.add_argument('--nondefault-tv-weights', nargs='+', default=[], type=float, 
                      help='tv weights for the style names specified (default: %(default)s)')

  def add_precomputed_features_argument(self):
    self.add_argument('--precomputed-features', default=False, help='train data points to images and precomputed features', action='store_true')

  def add_data_pipeline_arguments(self):
    self.add_argument('--train-path', type=str, help='train data path if precomputed is false, else precomputed features path', required=True)
    self.add_argument('--train-image-size', default=None, type=int, help='size of training images after resizing (default: %s)' % TRAIN_IMAGE_SIZE)
    self.add_argument('--subset-size', default=None, type=int, help='size of subset of training data (default: %(default)s)')
    self.add_argument('--batch-size', default=4, type=int, help='batch size (default %(default)s)')

  def add_style_transfer_network_arguments(self):
    self.add_argument('--conv-separable', default=False, help='use depthwise separable convolution', action='store_true')
    self.add_argument('--nonlinearity', default='relu', type=str, help='nonlinearity to use (default %(default)s)', choices=['relu', 'selu'])
    self.add_argument('--resize-factor', default=0, type=int, help='factor of resize in encoder decoder (default: %(default)s)', choices=[0,1,2])
    self.add_argument('--no-decoder-norm', default=True, help='no conditional instance normalization in decoder', action='store_false')
    self.add_argument('--variable-scope-name', default='style_transfer', type=str, help='name of variable scope of style transfer network')

  def add_train_step_arguments(self):
    self.add_argument('--learning-rate', default=1e-3, type=float, help='learning rate (default %(default)s)')

  def add_slow_optimizer_arguments(self):
    self.add_argument('--initial-image', default='content', type=str, help='initial image one of (noise, content, path to initial image)')
    self.add_argument('--total-iterations', default=200, type=int, help='number of iterations (default %(default)s)')
    self.add_argument('--checkpoint-iterations', default=5, type=int, help='checkpoint frequency (default %(default)s)')

  def add_fast_optimizer_arguments(self):
    self.add_argument('--checkpoint-iterations', default=500, type=int, help='checkpoint frequency (default: %(default)s)')
    self.add_argument('--total-epochs', default=4, type=int, help='number of epochs to run (default: %(default)s)')

  def add_layerwise_exps_arguments(self):
    self.add_argument('--init-loss', default=None, type=float, help='initial loss to start training(default: %(default)s)')

  def add_layerwise_multiple_exps_arguments(self):
    self.add_argument('--multiple-style-weights', nargs='+', default=[], type=float, help='style weights to use per layer (default: %(default)s)')

  def add_compute_features_arguments(self):
    self.add_argument('--content-layers', nargs='+', default=[], type=str, 
                      help='content layers for finding the content loss (default: as specified in loss_network_factory.py)')


class _OptionsChecker:
  def __init__(self, options):
    self.options = options

  def check_content_arguments(self):
    exists(self.options.content_path, "content path not found!")
    if(self.options.content_image_size is not None):
      assert self.options.content_image_size >= 128

  def check_result_arguments(self):
    assert not(os.path.exists(self.options.result_path)), "result dir already exists!"

  def check_test_arguments(self):
    if self.options.test_path: exists(self.options.test_path, "test images not found!")

  def check_style_arguments(self):
    exists(self.options.style_path, "style path not found!")
    if(self.options.style_image_size is not None):
      assert self.options.style_image_size >= 128

  def check_loss_network_arguments(self):
    loss_network_arch = self.options.loss_network.split('-')[0]
    return

  def check_loss_weights_arguments(self):
    assert self.options.content_weight >= 0
    if self.options.style_weight is not None:
      assert self.options.style_weight >= 0
    else:
      self.options.style_weight = DEFAULT_STYLE_WEIGHT
    assert self.options.tv_weight >= 0

  def check_layerwise_loss_arguments(self):
    assert len(self.options.content_layers)==len(set(self.options.content_layers))
    assert len(self.options.content_layers) >= len(self.options.layerwise_content_weights)
    if(len(self.options.layerwise_content_weights)>0):
      for layer_weight in self.options.layerwise_content_weights:
        assert layer_weight >= 0

    assert len(self.options.style_layers)==len(set(self.options.style_layers))
    assert len(self.options.style_layers) >= len(self.options.layerwise_style_weights)
    if(len(self.options.layerwise_style_weights)>0):
      for layer_weight in self.options.layerwise_style_weights:
        assert layer_weight >= 0

  def check_stylewise_loss_arguments(self):
    assert len(self.options.style_names)==len(self.options.nondefault_style_weights)
    assert len(self.options.style_names)==len(self.options.nondefault_tv_weights)
    if(len(self.options.style_names)>0):
      for style_weight in self.options.nondefault_style_weights:
        assert style_weight >= 0
      for tv_weight in self.options.nondefault_tv_weights:
        assert tv_weight >= 0

  def check_precomputed_features_argument(self):
    if(self.options.precomputed_features):
      assert self.options.train_image_size is None
    else:
      if(self.options.train_image_size is None):
        self.options.train_image_size = TRAIN_IMAGE_SIZE
      else:
        assert self.options.train_image_size >= 128

  def check_data_pipeline_arguments(self):
    exists(self.options.train_path, "training data not found!")
    assert (self.options.subset_size is None or self.options.subset_size >= 1000)
    assert self.options.batch_size > 0

  def check_compute_features_train_image_size_argument(self):
    if(self.options.train_image_size is None):
      self.options.train_image_size = TRAIN_IMAGE_SIZE
    else:
      assert self.options.train_image_size >= 128

  def check_style_transfer_network_arguments(self):
    return

  def check_train_step_arguments(self):
    assert self.options.learning_rate >= 0

  def check_slow_optimizer_arguments(self):
    if(self.options.initial_image not in ['content', 'noise']):
      exists(self.options.initial_image, "initial image not found!")
    assert self.options.total_iterations > 0
    assert self.options.checkpoint_iterations > 0

  def check_fast_optimizer_arguments(self):
    assert self.options.checkpoint_iterations > 0
    assert self.options.total_epochs > 0

  def check_layerwise_exps_arguments(self):
    assert len(self.options.style_layers)==1 or len(self.options.content_layers)==1
    if len(self.options.style_layers)==1:
      assert len(self.options.layerwise_style_weights)==0, "For a single layer, style-weight argument can be used"
    if len(self.options.content_layers)==1:
      assert len(self.options.layerwise_content_weights)==0, "For a single layer, content-weight argument can be used"
    if self.options.init_loss is not None:
      assert self.options.init_loss > 0

  def check_layerwise_multiple_exps_arguments(self):
    assert len(self.options.content_layers)==1
    if len(self.options.style_layers)==1:
      assert len(self.options.layerwise_style_weights)==0, "For a single layer, style-weight argument can be used"
    if self.options.init_loss is not None:
      assert self.options.init_loss > 0

    if len(self.options.multiple_style_weights) > 0:
      # All style weights must be unique
      assert len(self.options.multiple_style_weights)==len(set(self.options.multiple_style_weights))
      assert self.options.style_weight is None
      for style_wt in self.options.multiple_style_weights:
        assert style_wt >= 0
    else:
      if self.options.style_weight is not None:
        assert self.options.style_weight >= 0
      else:
        self.options.style_weight = DEFAULT_STYLE_WEIGHT

    assert self.options.content_weight >= 0
    assert self.options.tv_weight >= 0


def slow_optimizer_parser():
  parser = _Parser()
  parser.add_content_arguments()
  parser.add_result_arguments()
  parser.add_style_arguments()
  parser.add_loss_network_arguments()
  parser.add_loss_weights_arguments()
  parser.add_layerwise_loss_arguments()
  parser.add_slow_optimizer_arguments()
  return parser

def check_slow_optimizer_arguments(options):
  checker = _OptionsChecker(options)
  checker.check_content_arguments()
  checker.check_result_arguments()
  checker.check_style_arguments()
  checker.check_loss_network_arguments()
  checker.check_loss_weights_arguments()
  checker.check_layerwise_loss_arguments()
  checker.check_slow_optimizer_arguments()

def layerwise_exps_parser():
  parser = _Parser()
  parser.add_content_arguments()
  parser.add_result_arguments()
  parser.add_style_arguments()
  parser.add_loss_network_arguments()
  parser.add_loss_weights_arguments()
  parser.add_layerwise_loss_arguments()
  parser.add_slow_optimizer_arguments()
  parser.add_layerwise_exps_arguments()
  return parser

def check_layerwise_exps_arguments(options):
  checker = _OptionsChecker(options)
  checker.check_content_arguments()
  checker.check_result_arguments()
  checker.check_style_arguments()
  checker.check_loss_network_arguments()
  checker.check_loss_weights_arguments()
  checker.check_layerwise_loss_arguments()
  checker.check_slow_optimizer_arguments()
  checker.check_layerwise_exps_arguments()

def layerwise_multiple_exps_parser():
  parser = _Parser()
  parser.add_content_arguments()
  parser.add_result_arguments()
  parser.add_style_arguments()
  parser.add_loss_network_arguments()
  parser.add_loss_weights_arguments()
  parser.add_layerwise_loss_arguments()
  parser.add_slow_optimizer_arguments()
  parser.add_layerwise_exps_arguments()
  parser.add_layerwise_multiple_exps_arguments()
  return parser

def check_layerwise_multiple_exps_arguments(options):
  checker = _OptionsChecker(options)
  checker.check_content_arguments()
  #checker.check_result_arguments()
  checker.check_style_arguments()
  checker.check_loss_network_arguments()
  checker.check_layerwise_loss_arguments()
  checker.check_slow_optimizer_arguments()
  checker.check_layerwise_multiple_exps_arguments()

def fast_optimizer_parser():
  parser = _Parser()
  parser.add_result_arguments()
  parser.add_precomputed_features_argument()
  parser.add_data_pipeline_arguments()
  parser.add_style_arguments()
  parser.add_test_arguments()
  parser.add_style_transfer_network_arguments()
  parser.add_loss_network_arguments()
  parser.add_loss_weights_arguments()
  parser.add_layerwise_loss_arguments()
  parser.add_stylewise_loss_arguments()
  parser.add_train_step_arguments()
  parser.add_fast_optimizer_arguments()
  return parser

def check_fast_optimizer_arguments(options):
  checker = _OptionsChecker(options)
  checker.check_result_arguments()
  checker.check_precomputed_features_argument()
  checker.check_data_pipeline_arguments()
  checker.check_style_arguments()
  checker.check_test_arguments()
  checker.check_style_transfer_network_arguments()
  checker.check_loss_network_arguments()
  checker.check_loss_weights_arguments()
  checker.check_layerwise_loss_arguments()
  checker.check_stylewise_loss_arguments()
  checker.check_fast_optimizer_arguments()

def compute_features_parser():
  parser = _Parser()
  parser.add_loss_network_arguments()
  parser.add_data_pipeline_arguments()
  parser.add_result_arguments()
  parser.add_compute_features_arguments()
  return parser

def check_compute_features_arguments(options):
  checker = _OptionsChecker(options)
  checker.check_loss_network_arguments()
  checker.check_data_pipeline_arguments()
  checker.check_compute_features_train_image_size_argument()
  checker.check_result_arguments()

def create_layerwise_loss_weights_dict(layers, layerwise_weights, default_value=None):
  if(default_value is not None):
    assert len(layers) >= len(layerwise_weights)
    layerwise_weights = layerwise_weights + [default_value]*(len(layers) - len(layerwise_weights))
  layerwise_weights_dict = dict(zip(layers, layerwise_weights))
  return layerwise_weights_dict
