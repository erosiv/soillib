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

from scipy.ndimage import gaussian_filter

def calc_d8(data):

  # De-Normalize
  data = 2.0*data - 1.0
  data[:,:,1] *= -1
  direction = data[:,:,:2]

  # Get 2D Normal Vectors
  norm = np.sqrt(np.sum(direction**2, -1))
  norm = norm[..., np.newaxis]
  direction = direction / norm

  # Compute the Angle (Range [0, 1])
  d = np.arctan2(direction[:,:,1], direction[:,:,0])
  d = d / (2.0 * np.pi)
  d[d < 0] += 1.0

  # Compute the Clamped Directions (Anti-Clockwise f. East)
  d8 = np.full((norm.shape[0], norm.shape[1]), 0)
  d8[d >=  1.0 / 16.0] += 1
  d8[d >=  3.0 / 16.0] += 1
  d8[d >=  5.0 / 16.0] += 1
  d8[d >=  7.0 / 16.0] += 1
  d8[d >=  9.0 / 16.0] += 1
  d8[d >= 11.0 / 16.0] += 1
  d8[d >= 13.0 / 16.0] += 1
  d8[d >= 15.0 / 16.0] = 0
  return d8

def main():

  input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #input = "out_altmuenster.tiff"
  image = soil.tiff(input)
  node = image.node()

  print(f"File: {input}, {node.type}")

  normal = soil.normal(image.index, node)
  data = normal.full().numpy(image.index)

  # Temporary Gaussian
  # data = np.full((512, 512, 3), 1)
  # grid = np.indices((512, 512))
  # grid = grid - 256
  # data[:,:,0] = grid[0]
  # data[:,:,1] = grid[1]

  dir_d8 = calc_d8(data).astype(np.int64)

  # print(p)

  # Generate a Set of Paths...?
  # 1. Seed Random Points
  # 2. Step Based on D8 Position?
  # 3. Iterate this... store somehow
  # 4. See if we can visualize these paths somehow??
  # 5. That would basically confirm that we are doing it right.

  # Get original position...
  
  shape = dir_d8.shape
  samples = 5000

  pos = np.random.rand(samples, 2, 1)
  pos[:,0,:] *= shape[0]
  pos[:,1,:] *= shape[1]
  pos = pos.astype(np.int64)

  for n in range(512):
    # compute the next position
    npos = pos[..., -1].astype(np.int64)
    direction = dir_d8[npos[:,0], npos[:,1]]

    # For the Gaussian:
    npos[direction == 0] += np.array([ 1, 0])
    npos[direction == 1] += np.array([ 1, 1])
    npos[direction == 2] += np.array([ 0, 1])
    npos[direction == 3] += np.array([-1, 1])
    npos[direction == 4] += np.array([-1, 0])
    npos[direction == 5] += np.array([-1,-1])
    npos[direction == 6] += np.array([ 0,-1])
    npos[direction == 7] += np.array([ 1,-1])

    npos = np.clip(npos, [0,0], [shape[0]-1,shape[1]-1])
    pos = np.append(pos, npos[..., np.newaxis], axis=-1)

  fig, ax = plt.subplots()
  ax.set_xlim(0, shape[1])
  ax.set_ylim(0, shape[0])

  #plt.imshow(dir_d8)
  #plt.colorbar()

  pos = np.flip(pos, 1)
  line_collection = LineCollection(pos.transpose(0,2,1), color='black')
  ax.add_collection(line_collection)
  ax.scatter(pos[:,0,0], pos[:,1,0], color='black')

  plt.show()

if __name__ == "__main__":
  main()