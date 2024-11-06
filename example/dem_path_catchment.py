#!/usr/bin/env python

'''
Compute the Upstream Catchment Area for a DEM
The data is given in .TIFF format.

1. Compute Gradient / Surface Normal
2. Get Clamped D8 Map
3. Accumulate Upstream Nodes!
'''

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
  dirmap = (7, 8, 1, 2, 3, 4, 5, 6)
  flow_gt = grid.flowdir(dem, dirmap=dirmap)
  print("Computing Catchment...")
  area_gt = np.copy(grid.accumulation(flow_gt, dirmap=dirmap))

  # Load the Image, Get the Data

  image = soil.geotiff(input)
  raw_node = image.node()
  raw_data = raw_node.numpy(image.index)
  raw_mask = np.isnan(raw_data)

  print(f"File: {input}, {raw_node.type}, {raw_data.shape}")

  # Compute the Flow Direction

  print("Flow Node")

  flow_node = soil.flow(image.index, raw_node)
  flow = flow_node.full().numpy(image.index)
  print("Flow Difference", np.sum(flow_gt != flow))

#  print(type(flow_node.full()))
  
  dir_node = soil.direction(image.index, flow_node.full())
  direction = dir_node.full().numpy(image.index)
  shape = flow.shape

  print("Computing Area")

  area_node = soil.accumulation(image.index, dir_node.full())
  area = area_node.full().numpy(image.index)

  sum_dist = np.sum(np.abs(area_gt-area))
  dist_sum = np.abs(np.sum(area_gt) - np.sum(area))

  print(f"{sum_dist/np.sum(area_gt):.5f}, {dist_sum/np.sum(area_gt):.5f}")


  plt.imshow(np.abs(area - area_gt)/area_gt)
  plt.show()

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

  # Set of Valid Area Values
  # Number of Occurences corresponds in this array
  vals = np.sort(area_gt[flow > 0])

  # Set of Unique Values gives us an x coordinate
  # The index is the first occurence in the sorted array

  x, index = np.unique(vals, return_index=True)
  
  # Therefore, the total number of values is the total
  # "area" of the domain, the number of elements
  # The count is the number of occurences for each unique
  # element. We now have a map count(area_value).

  index_max = len(vals)
  count = np.append(index, index_max)[1:] - index

  # Scaling the count by the value, we get the total
  # contribution to the accumulation.
  # We then accumlate 

  area_t = count * x
  t = np.cumsum(area_t)

  ax[1,0].plot(np.log(x), np.log(t))

  # do the same thing again!

  vals = np.sort(area[flow > 0])

  # Set of Unique Values gives us an x coordinate
  # The index is the first occurence in the sorted array

  x, index = np.unique(vals, return_index=True)
  
  # Therefore, the total number of values is the total
  # "area" of the domain, the number of elements
  # The count is the number of occurences for each unique
  # element. We now have a map count(area_value).

  index_max = len(vals)
  count = np.append(index, index_max)[1:] - index

  # Scaling the count by the value, we get the total
  # contribution to the accumulation.
  # We then accumlate 

  area_t = count * x
  t = np.cumsum(area_t)

  ax[1,0].plot(np.log(x), np.log(t))

#  def _p(ax, area):
#    
#    vals = np.log(vals)
#  #  vals = np.floor(50*vals)/50
#
#    
#
#    count = np.append(index, len(vals))
#    count = count[1:] - index
#    ax.plot(x, np.log(count))
#
#  _p(ax[1,0], area_gt)
#  _p(ax[1,0], area)

#  vals = np.sort(area[flow > 0])
#  x, index = np.unique(vals, return_index=True)
#
#  count = np.append(index, len(vals))
#  count = count[1:] - index
#  ax[1,0].plot(np.log(x), np.log(count))

#  vals = np.sort(area[mask])
#  X, F = np.unique(vals, return_index=True)
#  ax[1,0].plot(np.log10(X), F)

#  counts, bins, = np.histogram(np.log10(vals), bins=512)
#  ax[1,0].stairs(np.log10(counts), bins)

  plt.show()

  '''
  mask = (flow > 0)
  area_gt = area_gt[mask]
  area = area[mask]

  counts, bins, = np.histogram(np.log(area), bins=32)
  ax[1, 0].stairs(np.log(counts), bins)

  plt.tight_layout()
  plt.show()
  '''

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Downloads/elevation.tiff"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #data = "out_altmuenster.tiff"
  #data = "/home/nickmcdonald/Datasets/elevation_conditioned.tiff"
  data = "conditioned.tiff"

  main(data)