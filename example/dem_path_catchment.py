#!/usr/bin/env python

'''
Compute the Upstream Catchment Area for a DEM
The data is given in .TIFF format.

1. Compute Gradient / Surface Normal
2. Get Clamped D8 Map
3. Accumulate Upstream Nodes!
'''

import os
import soillib as soil
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
import numpy as np
from tqdm import tqdm

dirmap = (7, 8, 1, 2, 3, 4, 5, 6)
coords = [
  np.array([ 0,-1]),
  np.array([ 1,-1]),
  np.array([ 1, 0]),
  np.array([ 1, 1]),
  np.array([ 0, 1]),
  np.array([-1, 1]),
  np.array([-1, 0]),
  np.array([-1,-1]),
]

def calc_d8(data):

  slope_stack = np.full((8, data.shape[0], data.shape[1]), 0.0)
  for k, coord in enumerate(coords):
    dist = np.sqrt(coord[0]**2 + coord[1]**2)
    slope_stack[k] = (data - np.roll(data, (-coord[0], -coord[1]), axis=(1,0)))/dist

  d8 = np.argmax(slope_stack, axis=0)
  d8 = np.asarray(list(dirmap))[d8]
  return d8

def main(input = ""):

  # Load the Image, Get the Data

  image = soil.geotiff(input)
  raw_node = image.node()
  raw_data = raw_node.numpy(image.index)

  print(f"File: {input}, {raw_node.type}, {raw_data.shape}")

  # Compute the Flow Direction

  print("Computing Direction...")

  dir_d8 = calc_d8(raw_data)
  shape = dir_d8.shape

  # We should just compute an actual direction map...
  direction = np.full((shape[0], shape[1], 2), 0)
  direction[dir_d8 == dirmap[0]] = coords[0]
  direction[dir_d8 == dirmap[1]] = coords[1]
  direction[dir_d8 == dirmap[2]] = coords[2]
  direction[dir_d8 == dirmap[3]] = coords[3]
  direction[dir_d8 == dirmap[4]] = coords[4]
  direction[dir_d8 == dirmap[5]] = coords[5]
  direction[dir_d8 == dirmap[6]] = coords[6]
  direction[dir_d8 == dirmap[7]] = coords[7]

  print("Computing Area")

  area = np.full(shape, 0)

  iterations = 256
  samples = 4096
  steps = 8192

  for i in range(iterations):
    print(f"ITERATION {i}")
    
    pos = np.random.rand(samples, 2)
    pos[..., 0] *= shape[0]
    pos[..., 1] *= shape[1]
    pos = pos.astype(np.int64)

    for n in tqdm(range(steps)):
      pos += direction[pos[:,1], pos[:,0]]
      pos = np.clip(pos, [0,0], [shape[0]-1,shape[1]-1])
      area[pos[:,1], pos[:,0]] += 1

  print("Plotting Accumulation...")

  area = area / (iterations*samples) * shape[0]*shape[1]
  area += 1

  area = area[1:-2,1:-2]

  fig = plt.figure(figsize=(8,6))
  fig.patch.set_alpha(0)
  ax = fig.add_subplot(1, 1, 1) 

  im = ax.imshow(area, zorder=2,
    cmap='cubehelix',
    norm=colors.LogNorm(1, area.max()),
    interpolation='bilinear')

  plt.colorbar(im, ax=ax, label='Upstream Cells')
  plt.grid(zorder=-1)
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