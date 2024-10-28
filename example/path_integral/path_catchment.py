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
import numpy as np

def clamp_dirs(data):

  # De-Normalize
  data = 2.0*data - 1.0
  direction = data[:,:,:2]

  # Get 2D Normal Vectors
  norm = np.sqrt(np.sum(direction**2, -1))
  norm = norm[..., np.newaxis]
  direction = direction / norm

  # Show 2D Direction
  # direction = 0.5 + 0.5*direction
  # direction = np.append(direction, np.full(norm.shape, 0), axis=2)
  # return direction

  # Compute the D8 Direction

  #d8[direction[:,:,0] >= 0] += 1
  #d8[direction[:,:,1] >= 0] += 1

  d = np.arctan2(direction[:,:,1], direction[:,:,0])
  d = d / (2.0 * np.pi)
  d[d < 0] += 1.0

  #print(d.shape)
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
  input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
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

  new_data = clamp_dirs(data)

  new_data = np.transpose(new_data, (1,0))
  plt.imshow(new_data)
  plt.colorbar()
  plt.show()

if __name__ == "__main__":
  main()