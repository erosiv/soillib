#!/usr/bin/env python

from __common__ import *

import soillib as soil
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as skt

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
    geotiff.peek(path)
    _meta = geotiff.meta
    if meta == None and _meta.gdal_metadata != "":
      meta = _meta

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
  mshape = soil.index([pixels[1], pixels[0]])

  print(f"Output Format: ({pixels[0]}, {pixels[1]})")

  array = soil.buffer(soil.float32, mshape.elem())
  soil.set(array, np.nan)

  for file, path in iter_tiff(input):

    print(f"Merging: {file}")
    with soil.timer(soil.ms) as timer:

      # Load the Geotiff, Get the Buffer in Numpy, Downscale
      geotiff = soil.geotiff(path)

#      data = geotiff.buffer.numpy(geotiff.index)
#      data = skt.rescale(data, pscale, anti_aliasing=True)
  
      # Get the World-Space Position of the Image
      gmin = np.array(geotiff.min)
      gmax = np.array(geotiff.max)
      gscale = np.array(geotiff.scale)
      soil.copy(array, geotiff.buffer, gmin, gmax, gscale, wmin, wmax, wscale, pscale)

  return array, mshape, meta

def main(input, file_out):

  array, shape, meta = merge(input, pscale=0.1)

  '''
  Figure out how to export this is a valid GeoTIFF!
  '''

  tiff_out = soil.geotiff(array, shape)
  tiff_out.meta = meta
  tiff_out.unsetnan()
  tiff_out.write(file_out)

  #show_relief(array, shape)
  #show_normal(array, shape)
  show_height(array, shape)

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/elevation.tiff"

  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40705_DGM_tif_Gmunden"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40704_DGM_tif_Ebensee"

  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/41225_DGM_tif_Ried"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/41709_DGM_tif_Frankenburg"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40706_DGM_tif_Gosau"
  data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40702_DGM_tif_Bad_Goisern"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40702_DGM_tif_Bad_Goisern/G-T4728-18.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/41234_DGM_tif_Waldzell"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/41403_DGM_tif_Brunnenthal"

  file_out = "_dem_merged.tiff"
  main(data, file_out)