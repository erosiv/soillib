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

coords = [
  np.array([ 0,-1]), # N
  np.array([ 1,-1]), # NE
  np.array([ 1, 0]), # E
  np.array([ 1, 1]), # SE
  np.array([ 0, 1]), # S
  np.array([-1, 1]), # SW
  np.array([-1, 0]), # W
  np.array([-1,-1]), # NW
]

def _flow(data):

  slopes = np.full((8, data.shape[0], data.shape[1]), 0.0)
  for k, coord in enumerate(coords):
    distance = np.sqrt(coord[0]**2 + coord[1]**2)
    slopes[k] = (data - np.roll(data, (-coord[0], -coord[1]), axis=(0,1)))/distance

  flow = np.argmax(slopes, axis=0)

  flow = np.asarray(list(dirmap))[flow]
  mask = np.isnan(data)
  flow[mask] = 0

  mask = np.less_equal(slopes, 0).all(axis=0)
  flow[mask] = 0

  '''
  print("FLOW IS ZERO")
  print(flow.shape)
  print(flow)

  # What if for some reason we stil have a pit?

  print(np.sum())

#  print(slopes.shape)
#  print(slopes[:,flow])
  '''

  return flow

def _direction(flow):

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

  iterations = 1024
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

 # plt.imshow(np.log(area_gt))
 # plt.show()

  # Load the Image, Get the Data

  image = soil.geotiff(input)
  raw_node = image.node()
  raw_data = raw_node.numpy(image.index)
  raw_mask = np.isnan(raw_data)

  print(f"File: {input}, {raw_node.type}, {raw_data.shape}")

  # Compute the Flow Direction

  print("Computing Direction...")

  flow = np.copy(flow_gt)
  #flow = _flow(raw_data)
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

  mask = (flow > 0)
  area_gt = area_gt[mask]
  area = area[mask]

  counts, bins, = np.histogram(np.log(area_gt), bins=32)
  ax[1, 0].stairs(np.log(counts), bins)

  counts, bins, = np.histogram(np.log(area), bins=32)
  ax[1, 0].stairs(np.log(counts), bins)

  plt.tight_layout()
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