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

  # Generate a Set of Paths...?
  # 1. Seed Random Points
  # 2. Step Based on D8 Position?
  # 3. Iterate this... store somehow
  # 4. See if we can visualize these paths somehow??
  # 5. That would basically confirm that we are doing it right.

  print("Computing Direction...")

  dir_d8 = calc_d8(raw_data)
  shape = dir_d8.shape
  
  print("Sampling Positions...")
  samples = 4096#*64
  steps = 4096
  pos = np.full((samples, 2, steps+1), 0.0)
  pos[..., 0] = np.random.rand(samples, 2)
  pos[..., 0, 0] *= shape[0]
  pos[..., 1, 0] *= shape[1]
  pos = pos.astype(np.int64)
 
  print("Generating Paths...")

  for n in tqdm(range(steps)):

    npos = pos[..., n]
    direction = dir_d8[npos[:,1], npos[:,0]]

    npos[direction == dirmap[0]] += coords[0]
    npos[direction == dirmap[1]] += coords[1]
    npos[direction == dirmap[2]] += coords[2]
    npos[direction == dirmap[3]] += coords[3]
    npos[direction == dirmap[4]] += coords[4]
    npos[direction == dirmap[5]] += coords[5]
    npos[direction == dirmap[6]] += coords[6]
    npos[direction == dirmap[7]] += coords[7]

    npos = np.clip(npos, [0,0], [shape[0]-1,shape[1]-1])
    pos[..., n+1] = npos

  print("Plotting Paths...")

  fig = plt.figure(figsize=(8,6))
  fig.patch.set_alpha(0)
  ax = fig.add_subplot(1, 1, 1) 

  boundaries = ([-2,-1,0,1,2,3,4,5,6,7,8])
  plt.imshow(1+dir_d8, cmap='viridis', zorder=2, vmin=-2, vmax=8)
  plt.colorbar(boundaries= boundaries)
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.title('Flow direction grid', size=14)
  plt.grid(zorder=-1)
  plt.tight_layout()

  line_collection = LineCollection(pos.transpose(0,2,1), color='black', alpha=1.0)
  ax.add_collection(line_collection)

  plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Downloads/elevation.tiff"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #data = "out_altmuenster.tiff"
  data = "/home/nickmcdonald/Datasets/elevation_conditioned.tiff"

  main(data)