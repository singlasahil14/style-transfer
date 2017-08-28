from optimize import SlowLossMinimizer
from argument_parser import *

def _find_loss_weight(loss, weight, init_loss=None):
  if(init_loss is not None and loss<init_loss):
    weight = init_loss/loss
  return weight

def main():
  parser = layerwise_exps_parser()
  options = parser.parse_args()
  check_layerwise_exps_arguments(options)

  layerwise_content_weights_dict = create_layerwise_loss_weights_dict(options.content_layers, options.layerwise_content_weights, default_value=1.)
  layerwise_style_weights_dict = create_layerwise_loss_weights_dict(options.style_layers, options.layerwise_style_weights, default_value=1.)

  minimizer = SlowLossMinimizer()
  minimizer.setup_feature_fn(model_identifier=options.loss_network, pool_layer=options.pool_layer, padding=options.padding, 
    content_layers=options.content_layers, style_layers=options.style_layers)
  minimizer.compute_content_style_features(options.content_path, options.style_path, 
    content_image_size=options.content_image_size, style_image_size=options.style_image_size)
  minimizer.setup_compute_graph(initial_identifier=options.initial_image)
  minimizer.setup_loss(options.content_weight, options.style_weight, options.tv_weight)
  content_val_dict, style_val_dict = minimizer.calc_all_layer_losses(print_losses=True)

  os.makedirs(options.result_path)
  content_layers, style_layers = minimizer.content_layers, minimizer.style_layers
  if(len(content_layers)==1):
    for style_layer in style_layers:
      result_path = os.path.join(options.result_path, style_layer)
      init_loss = style_val_dict[style_layer]
      style_layer_weight = layerwise_style_weights_dict[style_layer] if style_layer in layerwise_style_weights_dict else 1.
      style_layer_weight = _find_loss_weight(init_loss, style_layer_weight, options.init_loss)
      curr_layerwise_style_weights_dict = {style_layer: style_layer_weight}
      minimizer.setup_loss(options.content_weight, options.style_weight, options.tv_weight,
        layerwise_style_weights_dict=curr_layerwise_style_weights_dict, default_layerwise_style_weight=0.)
      minimizer.run_optimization(total_iterations=options.total_iterations, checkpoint_iterations=options.checkpoint_iterations, 
        result_path=result_path)
  elif(len(style_layers)==1):
    for content_layer in content_layers:
      result_path = os.path.join(options.result_path, content_layer)
      init_loss = content_val_dict[content_layer]
      content_layer_weight = layerwise_content_weights_dict[content_layer] if content_layer in layerwise_content_weights_dict else 1.
      content_layer_weight = _find_loss_weight(init_loss, content_layer_weight, options.init_loss)
      curr_layerwise_content_weights_dict = {content_layer: content_layer_weight}
      minimizer.setup_loss(options.content_weight, options.style_weight, options.tv_weight,
        layerwise_content_weights_dict=curr_layerwise_content_weights_dict, default_layerwise_content_weight=0.)
      minimizer.run_optimization(total_iterations=options.total_iterations, checkpoint_iterations=options.checkpoint_iterations, 
        result_path=result_path)
  else:
    raise ValueError('Either content_layers or style layers must be of size one')
  print("Experiment complete.")

if __name__ == '__main__':
  main()
