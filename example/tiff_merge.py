#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as skt

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
    yield file, path.decode('utf-8')

  elif os.path.isdir(path):
    for file in os.listdir(path):
      yield file, os.path.join(path, file).decode('utf-8')
      #return

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
  meta = None

  for file, path in iter_tiff(input):
  
    # Get Geotiff / Metadata

    geotiff = soil.geotiff()
    geotiff.meta(path)
    _meta = geotiff.get_meta()
    if meta == None and _meta.metadata != "":
      meta = _meta
      #print(meta.metadata)

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
  mshape = soil.index(pixels)

  array = soil.buffer(soil.float32, mshape.elem())
  array.fill(np.nan)

  for file, path in iter_tiff(input):

    print(f"Merging: {file}")
    with soil.timer(soil.ms) as timer:

      # Load the Geotiff, Get the Buffer in Numpy, Downscale
      geotiff = soil.geotiff(path)
      buf = geotiff.buffer()
      buf = buf.numpy().reshape((geotiff.height, geotiff.width))
      buf = skt.rescale(buf, pscale, anti_aliasing=True)
  
      # Get the World-Space Position of the Image
      gmin = np.array(geotiff.min)
      gmax = np.array(geotiff.max)
      gscale = np.array(geotiff.scale)
      
      # Get the Pixel-Space Extent of the Image
      pmin = (pscale * (gmin - wmin) / wscale).astype(np.int64)
      pmax = (pscale * (gmax - wmin) / wscale).astype(np.int64)
      shape = buf.shape

      # Re-Sample the Image
      for x in range(pmin[0], pmax[0]):
        for y in range(pmin[1], pmax[1]):
          index = x*pixels[1] + (pixels[1] - y - 1)
          px = int((x - pmin[0]))
          py = int((pmax[1]-y-1))
          array[index] = buf[py, px]

  return array, mshape, meta

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

  array, shape, meta = merge(input, pscale=0.2)

  '''
  Figure out how to export this is a valid GeoTIFF!
  '''

  tiff_out = soil.geotiff(array, shape)
  tiff_out.set_meta(meta)
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
  data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"

  main(data)