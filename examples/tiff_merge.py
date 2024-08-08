#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def show_array(array):
  data = np.array(array)
  data = np.transpose(data)
  plt.imshow(data)
  plt.show()

def main(path):

  # Check Path Validity

  path = os.fsencode(path)
  if not os.path.exists(path):
    raise RuntimeError("path does not exist")

  # Single-File Handling

  if os.path.isfile(path):
    tpath = path
    geotiff = soil.geotiff()
    geotiff.meta(tpath)
    geotiff.read(tpath)
    show_array(geotiff.buf())
    return

  # Determine World-Space Range of Data

  if not os.path.isdir(path):
    raise RuntimeError("path must be file or directory")

  wmin = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max])
  wmax = np.array([np.finfo(np.float32).min, np.finfo(np.float32).min])
  wscale = None

  for file in os.listdir(path):
  
    # Get Geotiff / Metadata

    tpath = os.path.join(path, file)
    geotiff = soil.geotiff()
    geotiff.meta(tpath)

    gmin = np.array(geotiff.min)
    gmax = np.array(geotiff.max)
    gscale = np.array(geotiff.scale)
  
    # Update Bounds Information

    wmin = np.min([wmin, gmin], axis=0)
    wmax = np.max([wmax, gmax], axis=0)
    wscale = gscale

  # Determine Merged Image Size

  '''
  # TODO make sure that we can do this with the array type.

  array.zero()
  '''

  '''


  array = np.empty(pixels)
  array[:] = np.nan

  '''

  pscale = 0.1
  pixels = pscale * ((wmax - wmin)/wscale)
  pixels = pixels.astype(np.int64)

  array = soil.array("float", pixels)
  array.fill(np.nan)
  # array = np.array(array)
  # print(array.shape)

  for file in os.listdir(path):

    tpath = os.path.join(path, file)
    geotiff = soil.geotiff(tpath)
    # print(tpath)

    buf = geotiff.array()
    buf = np.array(buf)
  
    gmin = np.array(geotiff.min)
    gmax = np.array(geotiff.max)
    gscale = np.array(geotiff.scale)
    
    pmin = (pscale * (gmin - wmin) / wscale).astype(np.int64)
    pmax = (pscale * (gmax - wmin) / wscale).astype(np.int64)

    # print(pmin, pmax)
    # print(pmax - pmin)
    # print(geotiff.width, geotiff.height)

    '''
    if we actually kept a shape variant inside the
    array type, we could implement the lookup with
    a visitor? that could potentially work.
    That would be better statically.
    '''

    shape = buf.shape

    print(f"Merging: {file}")
    with soil.timer() as timer:

      for x in range(pmin[0], pmax[0]):
        for y in range(pmin[1], pmax[1]):
          px = int((x - pmin[0]) / pscale)
          #px = int((pmax[0]-x-1) / pscale)
          #py = int((y - pmin[1]) / pscale)
          py = int((pmax[1]-y-1) / pscale)

          # TODO implement a more efficient indexing procedure here.
          # TODO we could actually fully implement the shape thing with visitors.
          # TODO figure why this is flipped. seems silly.

          array[x, pixels[1] - y - 1] = buf[py, px]

  show_array(array)

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster"
  main(data)