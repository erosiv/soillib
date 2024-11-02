import os
import soillib as soil
import matplotlib.pyplot as plt

def iter_tiff(path):

  '''
  Generator for all Files in 
  Directory, or a Single File
  '''

  path = os.fsencode(path)
  if not os.path.exists(path):
    raise RuntimeError("path does not exist")

  if os.path.isfile(path):
    file = os.path.basename(path)
    yield file, path.decode('utf-8')

  elif os.path.isdir(path):
    for file in os.listdir(path):
      yield file, os.path.join(path, file).decode('utf-8')

  else:
    raise RuntimeError("path must be file or directory")

def relief_shade(h, n):

  # Regularize Height
  h_min = np.nanmin(h)
  h_max = np.nanmax(h)
  h = (h - h_min)/(h_max - h_min)

  # Light Direction, Diffuse Lighting
  light = np.array([ 1, 1, 2])
  light = light / np.linalg.norm(light)

  diffuse = np.sum(light * n, axis=-1)
  diffuse = 0.05 + 0.9*diffuse

  # Flat-Toning
  flattone = np.full(h.shape, 0.9)
  weight = 1.0 - n[:,:,2]
  weight = weight * (1.0 - h * h)

  # Full Diffuse Shading Value
  diffuse = (1.0 - weight) * diffuse + weight * flattone
  return diffuse

def show_height(array, index):

  array = soil.cached(array)
  data = array.numpy(index)
  plt.imshow(data)
  plt.show()

def show_normal(array, index):

  array = soil.cached(array)
  normal = soil.normal(index, array)

  data = normal.full().numpy(index)
  plt.imshow(data)
  plt.show()

def show_relief(array, index):

  array = soil.cached(array)
  normal = soil.normal(index, array)

  normal_data = normal.full().numpy(index)
  height_data = array.numpy(index)
  
  relief = relief_shade(height_data, normal_data) 
  plt.imshow(relief, cmap='gray')
  plt.show()