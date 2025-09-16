#!/usr/bin/env python

import soillib as soil
import silt

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

'''
Example Code Showcasing Various
DEM Processing Kernels
'''

def load(data):

  image = soil.geotiff(data)
  print(f"Loaded File: {data}, {image.buffer.type}, ({image.index[0]}, {image.index[1]})")
  return (image.buffer, image.index)

def main(data):

  # Load GeoTIFF Data
  tiff = soil.geotiff(data)
  tensor = tiff.tensor.gpu()
  shape = tensor.shape
  res = (shape[0], shape[1])

  # Compute Accumulation
  rain = np.full(res, 1.0)
  rain = silt.tensor.from_numpy(rain.astype(np.float32)).gpu()

  t = soil.timer(soil.us)
  with t:
    dirn = soil.direction(tensor, soil.d8)
    flow = soil.steepest(tensor, soil.d8)
    accumulation = soil.accumulate(flow, rain, soil.d8)
  print(f"Execution Time: {t.count} us")

#  return

  accumulation = accumulation.cpu().numpy()  
  plt.imshow(accumulation,
    cmap='CMRmap',
    norm=colors.LogNorm(1, accumulation.max()),
    interpolation='none')

  plt.show()

  '''
  area[catch == 0] = 1
  plt.imshow(area, zorder=2,
    cmap='CMRmap',
    norm=colors.LogNorm(1, area.max()),
    interpolation='bilinear')
  plt.show()

  height = buffer.cpu().numpy(index)
  height[catch == 0] = np.nan
  tiff = soil.tiff(soil.buffer.from_numpy(height), index)
  tiff.write("height_masked.tiff")

  print("Plotting Data...")

  fig, ax = plt.subplots(2, 2, figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)

  im = ax[0, 0].imshow(flow, zorder=2,
    cmap='CMRmap',
    interpolation='bilinear')

  im = ax[0, 1].imshow(area, zorder=2,
    cmap='CMRmap',
    norm=colors.LogNorm(1, area.max()),
    interpolation='bilinear')

  im = ax[1, 0].imshow(catch, zorder=2,
    cmap='CMRmap',
    interpolation='none')

  im = ax[1, 1].imshow(distance, zorder=2,
    cmap='CMRmap',
    interpolation='bilinear')

  plt.show()
  '''

if __name__ == "__main__":

  data = "C:\\Users\\nicho\\Datasets\\test.tiff"
  #data = "_dem_conditioned.tiff"

  main(data)