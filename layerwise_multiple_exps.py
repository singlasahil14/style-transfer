from optimize import SlowLossMinimizer
from argument_parser import *

def _find_loss_weight(loss, weight, init_loss=None):
  if(init_loss is not None and loss<init_loss):
    weight = init_loss/loss
  return weight

def main():
  parser = layerwise_multiple_exps_parser()
  options = parser.parse_args()
  check_layerwise_multiple_exps_arguments(options)

  layerwise_content_weights_dict = create_layerwise_loss_weights_dict(options.content_layers, options.layerwise_content_weights, default_value=1.)
  layerwise_style_weights_dict = create_layerwise_loss_weights_dict(options.style_layers, options.layerwise_style_weights, default_value=1.)

  minimizer = SlowLossMinimizer()
  minimizer.setup_feature_fn(model_identifier=options.loss_network, pool_layer=options.pool_layer, padding=options.padding, 
    content_layers=options.content_layers, style_layers=options.style_layers)
  minimizer.compute_content_style_features(options.content_path, options.style_path, 
    content_image_size=options.content_image_size, style_image_size=options.style_image_size)
  minimizer.setup_compute_graph(initial_identifier=options.initial_image)

  if len(options.multiple_style_weights) == 0:
    multiple_style_weights = [options.style_weight]
  else:
    multiple_style_weights = options.multiple_style_weights
  multiple_style_weights.sort()
  multiple_style_weights.reverse()

  if not(os.path.exists(options.result_path)):
    os.makedirs(options.result_path)
  content_layers, style_layers = minimizer.content_layers, minimizer.style_layers
  continue_evaluation = dict(zip(style_layers, [True]*len(style_layers)))

  if(len(content_layers)==1):
    for style_weight in multiple_style_weights:
      minimizer.setup_loss(options.content_weight, style_weight, options.tv_weight)
      content_val_dict, style_val_dict = minimizer.calc_all_layer_losses(print_losses=True)
      for style_layer in style_layers:
        init_loss = style_val_dict[style_layer]
        style_layer_weight = layerwise_style_weights_dict[style_layer] if style_layer in layerwise_style_weights_dict else 1.
        style_layer_weight = _find_loss_weight(init_loss, style_layer_weight, options.init_loss)
        curr_layerwise_style_weights_dict = {style_layer: style_layer_weight}

        layer_result_path = os.path.join(options.result_path, style_layer)
        result_path = os.path.join(layer_result_path, 'style-weight-{}'.format(style_weight*style_layer_weight))

        if not(os.path.exists(result_path)) and continue_evaluation[style_layer]:
          minimizer.setup_loss(options.content_weight, style_weight, options.tv_weight,
            layerwise_style_weights_dict=curr_layerwise_style_weights_dict, default_layerwise_style_weight=0.)
          minimizer.run_optimization(total_iterations=options.total_iterations, checkpoint_iterations=options.checkpoint_iterations, 
            result_path=result_path)
        if options.init_loss is not None:
          if init_loss < options.init_loss:
            continue_evaluation[style_layer] = False
  else:
    raise ValueError('content_layers must be of length one')
  print("Experiment complete.")

if __name__ == '__main__':
  main()
