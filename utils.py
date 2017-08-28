import scipy.misc, numpy as np, os, sys
from shutil import rmtree, move
import tensorflow as tf
from PIL import Image
import glob, random, bcolz
from six.moves import urllib as six_urllib
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib
from collections import OrderedDict
import tarfile

_RESIZE_SIDE_MIN = 320

class OrderedDefaultListDict(OrderedDict):
    def __missing__(self, key):
      self[key] = value = []
      return value

def _central_crop(image, side):
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  offset_height = (image_height - side) / 2
  offset_width = (image_width - side) / 2

  original_shape = tf.shape(image)
  cropped_shape = tf.stack([side, side, 3])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
  image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)

def _smallest_size_at_least(height, width, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height, width = shape[0], shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image, [0])
  resized_image.set_shape([None, None, 3])
  return resized_image

def preprocess_image(image, side, resize_side=_RESIZE_SIDE_MIN, is_training=False):
  image = _aspect_preserving_resize(image, resize_side)
  if(is_training):
    image = tf.random_crop(image, [side, side, 3])
  else:
    image = _central_crop(image, side)
  return image

class data_pipeline:
  def __init__(self, train_path, sq_size=256, batch_size=4, subset_size=None):
    self.sq_size = sq_size

    filenames = glob.glob(train_path+'/*/*.jpg')
    random.shuffle(filenames)
    with tf.device('/cpu:0'):
      filenames = tf.constant(filenames[:subset_size])
      dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)
      dataset = dataset.filter(self._filter_function)
      dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
      dataset = dataset.batch(batch_size)
      self.iterator = dataset.make_initializable_iterator()

  def _filter_function(self, filename):
    image_string = tf.read_file(filename)
    img = tf.image.decode_jpeg(image_string)
    shp = tf.shape(img)
    return tf.logical_and(tf.equal(tf.rank(img), 3), tf.equal(shp[2], 3))

  def _parse_function(self, filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string)
    image = preprocess_image(image, self.sq_size, is_training=True)
    return image

def open_bcolz_arrays(root_dir, keys, arrs, mode='a', attr_dict={}):
  bcolz_arrs_dict = {}
  for key, arr in zip(keys, arrs):
    bcolz_path = os.path.join(root_dir, key)
    bcolz_arr = bcolz.carray(arr, rootdir=bcolz_path, mode=mode)
    for k,v in attr_dict.items():
      bcolz_arr.attrs[k] = v
    bcolz_arrs_dict[key] = bcolz_arr
  return bcolz_arrs_dict

class features_pipeline:
  def __init__(self, root_dir, keys, batch_size=16, attr_dict={}):
    bcolz_paths = [os.path.join(root_dir, key) for key in keys]
    with tf.device('/cpu:0'):
      bcolz_datasets = [self._bcolz_dataset(bcolz_path) for bcolz_path in bcolz_paths]
      dataset = tf.contrib.data.Dataset.zip(tuple(bcolz_datasets))
      dataset = dataset.batch(batch_size)
      self.iterator = dataset.make_initializable_iterator()

  def _bcolz_dataset(self, bcolz_path, attr_dict={}):
    arr = bcolz.open(rootdir=bcolz_path, mode='r')
    for k,v in attr_dict.items():
      assert arr.attrs[k]==v, "loss network mismatch"
    dataset = tf.contrib.data.Dataset.range(len(arr))
    py_func = lambda y: self._parse_function(y, arr)
    dataset = dataset.map(lambda x: tf.py_func(py_func, [x], [tf.float32]))
    dataset = dataset.map(lambda x: tf.reshape(x, arr.shape[1:]))
    return dataset

  def _parse_function(self, i, arr):
    elem = arr[i]
    return elem

def crop_and_resize(img, side=None):
  if(side==None):
    img = np.asarray(img)
    return img
  shortest = float(min(img.width,img.height))
  resized = np.round(np.multiply(side/shortest, img.size)).astype(int)
  img = img.resize(resized, Image.BILINEAR)
  left = (img.width - side)/2
  top = (img.height - side)/2
  img = np.asarray(img)
  img = img[top:top+side, left:left+side, :]
  return img

def save_image(out_path, img):
  img = np.clip(img, 0, 255).astype(np.uint8)
  scipy.misc.imsave(out_path, img)

def load_images(path, image_size=None):
  valid_exts = ['.jpeg', '.jpg']
  image_names = []
  images = []
  if(os.path.isdir(path)):
    for file_name in os.listdir(path):
      base, ext = os.path.splitext(file_name)
      if ext in valid_exts:
        image_names.append(base)
        image = Image.open(os.path.join(path, file_name))
        image = crop_and_resize(image, image_size)
        images.append(image)
    assert len(images) > 0
  elif(os.path.isfile(path)):
    folder_name, file_name = os.path.split(path)
    base, ext = os.path.splitext(file_name)
    assert ext in valid_exts
    image_names = [base]
    image = Image.open(os.path.join(path))
    image = crop_and_resize(image, image_size)
    images = [image]
  else:
    raise ValueError('Uninterpretable path')
  return image_names, images

def create_subfolder(super_folder, folder_name):
  new_folder_path = os.path.join(super_folder, folder_name)
  os.makedirs(new_folder_path)

def load_image(src, image_size=None):
  image = Image.open(os.path.join(src))
  image = crop_and_resize(image, image_size)
  return image

def exists(p, msg):
  assert os.path.exists(p), msg

def create_folder(dir_name, msg):
  assert not(os.path.exists(dir_name)), msg
  os.makedirs(dir_name)

def tensor_shape(tensor):
  shp = tf.shape(tensor)
  return shp[0], shp[1], shp[2], shp[3]

def download_model(model_url, model_dir, model_name):
  """Downloads the `model_url`, uncompresses it, saves the model file with
     the name model_name (default if unspecified) in the folder model_dir.

  Args:
    model_url: The URL of the model tarball file.
    model_dir: The directory where the model files are stored.
    model_name: The name of the model checkpoint file
  """
  model_name = model_name + '.ckpt'
  model_path = os.path.join(model_dir, model_name)
  if os.path.exists(model_path):
    return
  tmp_dir = os.path.join(model_dir, 'tmp')
  if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
  filename = model_url.split('/')[-1]
  filepath = os.path.join(tmp_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = six_urllib.request.urlretrieve(model_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(tmp_dir)

  ckpt_files = glob.glob(os.path.join(tmp_dir, '*.ckpt')) + glob.glob(os.path.join(tmp_dir, '*/*.ckpt'))
  assert len(ckpt_files)==1
  folder_name, file_name = os.path.split(ckpt_files[0])
  move(ckpt_files[0], os.path.join(model_dir, model_name))
  rmtree(tmp_dir)
