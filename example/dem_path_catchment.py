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
Flow and Direction Map Computation

Utilizes the Steepest Neighbor
to Compute D8 Flow Direction.
'''

dirmap = (7, 8, 1, 2, 3, 4, 5, 6)

# coords = [
#   np.array([ 0,-1]), # N
#   np.array([ 1,-1]), # NE
#   np.array([ 1, 0]), # E
#   np.array([ 1, 1]), # SE
#   np.array([ 0, 1]), # S
#   np.array([-1, 1]), # SW
#   np.array([-1, 0]), # W
#   np.array([-1,-1]), # NW
# ]

coords = [
    np.array([-1, 0]), # N
    np.array([-1, 1]), # NE
    np.array([ 0, 1]), # E
    np.array([ 1, 1]), # SE
    np.array([ 1, 0]), # S
    np.array([ 1,-1]), # SW
    np.array([ 0,-1]), # W
    np.array([-1,-1]), # NW
  ]


def _flow(data):

  slopes = np.full((8, data.shape[0], data.shape[1]), 0.0)
  for k, coord in enumerate(coords):
    distance = np.sqrt(coord[0]**2 + coord[1]**2)
    slopes[k] = (data - np.roll(data, (-coord[0], -coord[1]), axis=(0,1)))/distance

  has_flow = np.greater(slopes, 0.0).any(axis=0)
  slopes[np.isnan(slopes)] = -1.0

  flow = np.argmax(slopes, axis=0)
  flow = np.asarray(list(dirmap))[flow]

  pits = np.less(slopes, 0.0).all(axis=0)

  flat = ~has_flow & ~pits
  flow[~has_flow] = -1

  mask = np.isnan(data)
  flow[flat] = -1
  flow[pits] = -2
  flow[mask] = -2

  return flow

def _direction(flow):

  shape = flow.shape
  direction = np.full((shape[0], shape[1], 2), 0)
  direction[flow == dirmap[0]] = coords[0]
  direction[flow == dirmap[1]] = coords[1]
  direction[flow == dirmap[2]] = coords[2]
  direction[flow == dirmap[3]] = coords[3]
  direction[flow == dirmap[4]] = coords[4]
  direction[flow == dirmap[5]] = coords[5]
  direction[flow == dirmap[6]] = coords[6]
  direction[flow == dirmap[7]] = coords[7]

  return direction

def _area(height, flow, direction, area_gt):

  '''
  Iteratively Compute the Accumulation Area
  '''

  shape = height.shape
  area = np.full(shape, 0.0)
  count = np.full(shape, 0)

  iterations = 32
  samples = 1024
  steps = 3072

  #np.random.seed(0)

  for i in range(iterations):

    mask = np.full(samples, False)
    pos = np.random.rand(samples, 2)
    pos[..., 0] *= shape[0]
    pos[..., 1] *= shape[1]
    pos = pos.astype(np.int64)

    # Print Metrics

    P = (i * samples)/(shape[0] * shape[1])
    sum_dist = np.sum(np.abs(area_gt-area))
    dist_sum = np.abs(np.sum(area_gt) - np.sum(area))

    print(f"({i:3d}, {P:.5f}), {sum_dist/np.sum(area_gt):.5f}, {dist_sum/np.sum(area_gt):.5f}")

    # Count the Positions

    for n in range(steps):

      # Sample the Next Position, Check for Motion
      # Mask at Next Position if Out-Of-Bounds

      pos_next = pos + direction[pos[:,0], pos[:,1]]
      mask = np.logical_or(mask, pos_next[:,0] < 0)
      mask = np.logical_or(mask, pos_next[:,1] < 0)
      mask = np.logical_or(mask, pos_next[:,0] >= shape[0])
      mask = np.logical_or(mask, pos_next[:,1] >= shape[1])

      # Mask if Position has not Moved
      # Note: Implicit when flow <= 0, or height is nan
      mask = np.logical_or(mask, (pos_next == pos).all(axis=1))

      # Clip Position and Accumulate Value
      pos = np.clip(pos_next, [0,0], [shape[0]-1,shape[1]-1])
      np.add.at(count, (pos[:,0], pos[:,1]), ~mask)

    P = (shape[0] * shape[1])/((i+1)*(samples))
    area = 1.0 + P * count

  return area

'''
Main Control Flow
'''

def main(input = ""):

  print("Loading DEM for Ground-Truth")

  grid = Grid.from_raster(input)
  dem = grid.read_raster(input)

  print("Computing Flow...")
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

  print("Computing Direction...")

  flow = _flow(raw_data)

  print("Flow Difference", np.sum(flow_gt != flow))  
  
  direction = _direction(flow)
  shape = flow.shape

  print("Computing Area")

  area = _area(raw_data, flow, direction, area_gt)

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