from optimize import FastLossMinimizer
from argument_parser import *
from utils import *

def main():
  parser = compute_features_parser()
  options = parser.parse_args()
  check_compute_features_arguments(options)

  minimizer = FastLossMinimizer()
  minimizer.setup_feature_fn(model_identifier=options.loss_network, pool_layer=options.pool_layer, padding=options.padding, 
  	content_layers=options.content_layers)
  minimizer.compute_content_features(options.train_path, options.result_path, sq_size=options.train_image_size, batch_size=options.batch_size, 
    subset_size=options.subset_size)

if __name__ == '__main__':
  main()
