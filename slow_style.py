from optimize import SlowLossMinimizer
from argument_parser import *

def main():
  parser = slow_optimizer_parser()
  options = parser.parse_args()
  check_slow_optimizer_arguments(options)

  layerwise_content_weights_dict = create_layerwise_loss_weights_dict(options.content_layers, options.layerwise_content_weights)
  layerwise_style_weights_dict = create_layerwise_loss_weights_dict(options.style_layers, options.layerwise_style_weights)
  print layerwise_content_weights_dict, layerwise_style_weights_dict

  minimizer = SlowLossMinimizer()
  minimizer.setup_feature_fn(model_identifier=options.loss_network, pool_layer=options.pool_layer, padding=options.padding, 
    content_layers=options.content_layers, style_layers=options.style_layers)
  minimizer.compute_content_style_features(options.content_path, options.style_path, 
    content_image_size=options.content_image_size, style_image_size=options.style_image_size)
  minimizer.setup_compute_graph(initial_identifier=options.initial_image)
  minimizer.setup_loss(options.content_weight, options.style_weight, options.tv_weight,
    layerwise_content_weights_dict=layerwise_content_weights_dict, layerwise_style_weights_dict=layerwise_style_weights_dict)
  minimizer.calc_all_layer_losses(print_losses=True)
  minimizer.run_optimization(total_iterations=options.total_iterations, checkpoint_iterations=options.checkpoint_iterations, 
    result_path=options.result_path)
  print("Training complete.")

if __name__ == '__main__':
  main()
