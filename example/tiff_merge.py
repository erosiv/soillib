#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

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
    return file, path

  elif os.path.isdir(path):
    for file in os.listdir(path):
      yield file, os.path.join(path, file).decode('utf-8')

  else:
    raise RuntimeError("path must be file or directory")

def merge(input, pscale = 0.1):

  '''
  Generate a Merged Array from a Directory
  '''

  # Get World Extent

  wmin = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max])
  wmax = np.array([np.finfo(np.float32).min, np.finfo(np.float32).min])
  wscale = None

  for file, path in iter_tiff(input):
  
    # Get Geotiff / Metadata

    geotiff = soil.geotiff()
    geotiff.meta(path)

    gmin = np.array(geotiff.min)
    gmax = np.array(geotiff.max)
    gscale = np.array(geotiff.scale)
  
    # Update Bounds Information

    wmin = np.min([wmin, gmin], axis=0)
    wmax = np.max([wmax, gmax], axis=0)
    wscale = gscale
  
  # Determine Merged Image Size
  # Create Merged Filling Array
  
  pixels = (pscale * ((wmax - wmin)/wscale)).astype(np.int64)
  pixels_ = soil.index(pixels)
  array = soil.buffer(soil.float32, pixels_.elem())
  array.fill(np.nan)

  for file, path in iter_tiff(input):

    geotiff = soil.geotiff(path)
    buf = geotiff.buffer()
    buf = buf.numpy().reshape((geotiff.height, geotiff.width))
  
    gmin = np.array(geotiff.min)
    gmax = np.array(geotiff.max)
    gscale = np.array(geotiff.scale)
    
    pmin = (pscale * (gmin - wmin) / wscale).astype(np.int64)
    pmax = (pscale * (gmax - wmin) / wscale).astype(np.int64)
    shape = buf.shape

    print(f"Merging: {file}")
    with soil.timer(soil.ms) as timer:

      for x in range(pmin[0], pmax[0]):
        for y in range(pmin[1], pmax[1]):
          px = int((x - pmin[0]) / pscale)
          py = int((pmax[1]-y-1) / pscale)
          index = x*pixels[1] + (pixels[1] - y - 1)
          array[index] = buf[py, px]

  return array, pixels_

def show_height(array, index):

  array = soil.cached(array)
  data = array.numpy(index)
  data = np.transpose(data)
  plt.imshow(data)
  plt.show()

def show_normal(array, index):

  array = soil.cached(array)
  normal = soil.normal(index, array)

  data = normal.full().numpy(index)
  data = np.transpose(data, (1, 0, 2))
  plt.imshow(data)
  plt.show()

def show_relief(array, index):

  array = soil.cached(array)
  normal = soil.normal(index, array)

  normal_data = normal.full().numpy(index)
  height_data = array.numpy(index)
  
  relief = relief_shade(height_data, normal_data) 
  relief = np.transpose(relief, (1, 0))
  plt.imshow(relief, cmap='gray')
  plt.show()

def main(input):

  array, shape = merge(input, pscale=0.1)

  tiff_out = soil.tiff(array, shape)
  tiff_out.write("out.tiff")

  #show_relief(array, shape)
  show_normal(array, shape)
  #show_height(array, shape)

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40705_DGM_tif_Gmunden"
  data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40704_DGM_tif_Ebensee"
  main(data)