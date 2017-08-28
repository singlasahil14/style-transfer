from optimize import FastLossMinimizer
from argument_parser import *

def main():
  parser = fast_optimizer_parser()
  options = parser.parse_args()
  check_fast_optimizer_arguments(options)

  stylewise_style_weights_dict = dict(zip(options.style_names, map(float, options.nondefault_style_weights)))
  stylewise_tv_weights_dict = dict(zip(options.style_names, map(float, options.nondefault_tv_weights)))

  layerwise_content_weights_dict = create_layerwise_loss_weights_dict(options.content_layers, options.layerwise_content_weights)
  layerwise_style_weights_dict = create_layerwise_loss_weights_dict(options.style_layers, options.layerwise_style_weights)

  minimizer = FastLossMinimizer()
  minimizer.setup_feature_fn(model_identifier=options.loss_network, pool_layer=options.pool_layer, padding=options.padding, 
    content_layers=options.content_layers, style_layers=options.style_layers)
  minimizer.compute_style_features(options.style_path, style_image_size=options.style_image_size)
  if(options.precomputed_features):
    minimizer.setup_features_pipeline(options.train_path, batch_size=options.batch_size, subset_size=options.subset_size)
  else:
    minimizer.setup_data_pipeline(options.train_path, image_size=options.train_image_size, batch_size=options.batch_size, 
      subset_size=options.subset_size)
  minimizer.setup_compute_graph(nonlinearity=options.nonlinearity, decoder_norm=options.no_decoder_norm, resize_factor=options.resize_factor, 
    conv_separable=options.conv_separable, variable_scope_name=options.variable_scope_name)
  minimizer.setup_loss(options.content_weight, options.style_weight, options.tv_weight, 
    stylewise_style_weights_dict=stylewise_style_weights_dict, stylewise_tv_weights_dict=stylewise_tv_weights_dict, 
    layerwise_content_weights_dict=layerwise_content_weights_dict, layerwise_style_weights_dict=layerwise_style_weights_dict)
  minimizer.setup_train_step(learning_rate=options.learning_rate)
  minimizer.run_optimization(options.result_path, total_epochs=options.total_epochs, checkpoint_iterations=options.checkpoint_iterations, 
                             test_path=options.test_path)

  cmd_text = 'python evaluate.py --model-path %s ...' % model_path
  print("Training complete. For evaluation:\n    `%s`" % cmd_text)

if __name__ == '__main__':
  main()
