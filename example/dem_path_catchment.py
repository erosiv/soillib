#!/usr/bin/env python

import os
from pysheds.grid import Grid
import soillib as soil
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
import numpy as np
from tqdm import tqdm
from __common__ import *

'''
Main Control Flow
'''

def main(input = ""):

  print("Loading DEM for Ground-Truth")

  grid = Grid.from_raster(input)
  dem = grid.read_raster(input)

  print("Computing Flow...")
  #with soil.timer() as timer:
  dirmap = (7, 8, 1, 2, 3, 4, 5, 6)
  flow_gt = grid.flowdir(dem, dirmap=dirmap)
  
  print("Computing Catchment...")
  with soil.timer() as timer:
    area_gt = np.copy(grid.accumulation(flow_gt, dirmap=dirmap))

  # Load the Image, Get the Data

  image = soil.geotiff(input)
  print(f"File: {input}, {image.buffer.type}, ({image.index[0]}, {image.index[1]})")

  raw_node = soil.cached(image.buffer)
  raw_data = image.buffer.numpy(image.index)
  raw_mask = np.isnan(raw_data)

  # Compute the Flow Direction

  raw_node.buffer.gpu()

  print("Flow Node")
  with soil.timer() as timer:
    flow_node = soil.flow(image.index, raw_node.buffer)
  with soil.timer() as timer:
    dir_node = soil.direction(image.index, flow_node)
 
  print("Computing Area")
  area_node = soil.accumulation(image.index, dir_node)
  area_node.iterations = 64
  area_node.samples = 4096
  area_node.steps = 8192
  with soil.timer() as timer:
    area = area_node()
  area.cpu()

#  areas = []

#  print("Flow Difference", np.sum(flow_gt != flow))
#  print(type(flow_node.full()))
#  direction = dir_node.full().numpy(image.index)
#  shape = flow.shape

  area = area.numpy(image.index)
  sum_dist = np.sum(np.abs(area_gt-area))
  dist_sum = np.abs(np.sum(area_gt) - np.sum(area))

  print(f"{sum_dist/np.sum(area_gt):.5f}, {dist_sum/np.sum(area_gt):.5f}")

  fig, ax = plt.subplots(2, 2, figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)

  im = ax[0, 0].imshow(area_gt, zorder=2,
    cmap='CMRmap',
    norm=colors.LogNorm(1, area_gt.max()),
    interpolation='bilinear')
  #plt.colorbar(im, ax=ax[0], label='Upstream Cells')

  im = ax[0, 1].imshow(area, zorder=2,
    cmap='CMRmap',
    norm=colors.LogNorm(1, area.max()),
    interpolation='bilinear')
  #plt.colorbar(im, ax=ax[1], label='Upstream Cells')

  '''
  Area Histogram Plot: Because of the fractal nature,
  we should expect the connectivity of the graph to follow
  a power law. This means a log-log plot should be linear.

  Note that a minimum possible surface area estimate exists,
  which is when a single sample hits. = 1 * area / samples
  '''

  '''
  area_gt[raw_mask] = 0

  vals = np.sort(area_gt[area_gt >= 1])
  x, index = np.unique(vals, return_index=True)
  count = np.append(index, len(vals))[1:] - index
  cdf = np.cumsum(count*x)#/len(vals)
  ax[1,0].plot(np.log10(x), np.log(cdf))

  for k, val in enumerate([16, 32, 64]):#, 128, 256, 512, 1024]):
    area_node.iterations = val
    area = area_node().numpy(image.index)
    area[raw_mask] = 0

    vals = np.sort(area[area > 1])
    x, index = np.unique(vals, return_index=True)
    count = np.append(index, len(vals))[1:] - index
    cdf = np.cumsum(count*x)#/len(vals)
    ax[1,0].plot(np.log10(x), np.log(cdf))

  #vals = np.sort(area[area >= 1])
  #counts, bins, = np.histogram(np.log10(vals), bins=16)
  #ax[1,0].stairs(np.log10(counts), bins)
  '''

  plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Downloads/elevation.tiff"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #data = "out_altmuenster.tiff"
  #data = "/home/nickmcdonald/Datasets/elevation_conditioned.tiff"
  data = "conditioned.tiff"

  main(data)