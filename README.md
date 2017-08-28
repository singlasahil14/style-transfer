## style-transfer
style transfer code for generating stylized images using content and style loss from different layers, from different loss networks (vgg-16, vgg-19, inception-v1, inception-v2, inception-v3 trained on imagenet, inception-v3 trained on openimages, inception-v4

## Requirements
tensorflow r1.3

## Documentation
### Training Style Transfer Networks
```bash
./setup.sh
python fast_style.py --test-path data/content/ --train-path data/train/ --result-path test-fast/ --style-path data/style/wave.jpg
```
To access all command line arguments, run python fast_style.py --help

### Stylizing a single image
```bash
python slow_style.py --style-path data/style/wave.jpg --content-path data/content/stata.jpg --result-path test-slow/
```
To access all command line arguments, run python slow_style.py --help

### Printing model layers with tensor sizes
```bash
python print_model_layers.py --network inception-v4```
To access all command line arguments, run python print_model_layers.py --help

## Roadmap
- [ ] Add explanatory comments
- [ ] Expose more command-line arguments

## Contributing
Please feel free to:

* Create an issue
* Open a Pull Request
* Join the [gitter chat](https://gitter.im/style-transfer/Lobby)
* Share your success stories!
