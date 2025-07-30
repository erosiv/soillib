#!/usr/bin/env python

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

  for file, path in soil.util.iter_tiff(input):
  
    # Get Geotiff / Metadata

    geotiff = soil.geotiff()
    geotiff.peek(path)
    _meta = geotiff.meta
    if meta == None:# and _meta.gdal_metadata != "":
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
  mshape = soil.shape(pixels[1], pixels[0])

  print(f"Output Format: ({pixels[0]}, {pixels[1]})")

  array = soil.tensor(soil.float32, mshape)
  soil.set(array, np.nan)

  for file, path in soil.util.iter_tiff(input):

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
      soil.copy(array, geotiff.tensor, gmin, gmax, gscale, wmin, wmax, wscale, pscale)

  return array, mshape, meta

def main(input, file_out):

  array, shape, meta = merge(input, pscale=0.1)

  '''
  Figure out how to export this is a valid GeoTIFF!
  '''

  tiff_out = soil.geotiff(array)
  tiff_out.meta = meta
  tiff_out.unsetnan()
  tiff_out.write(file_out)

  #show_relief(array, shape)
  #show_normal(array, shape)
  soil.util.show_height(array)

if __name__ == "__main__":
  data = "data/dem_1024.tiff"
  file_out = "data/merged.tiff"
  main(data, file_out)