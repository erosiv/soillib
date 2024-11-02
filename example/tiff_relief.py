#!/usr/bin/env python

from __common__ import *

import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def relief_shade(h, n):

  # Regularize Height
  h_min = np.nanmin(h)
  h_max = np.nanmax(h)
  h = (h - h_min)/(h_max - h_min)

  # Light Direction, Diffuse Lighting
  light = np.array([-1, 2, 1])
  light = light / np.linalg.norm(light)

  n = 0.5*n + 0.5
  diffuse = np.sum(light * n, axis=-1)
  diffuse = 0.05 + 0.9*diffuse

  # Flat-Toning
  flattone = np.full(h.shape, 0.9)
  weight = 1.0 - n[:,:,2]
  weight = weight * (1.0 - h * h)

  # Full Diffuse Shading Value
  diffuse = (1.0 - weight) * diffuse + weight * flattone
  return diffuse

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    index = image.index

    height_node = image.node()
    normal_node = soil.normal(index, height_node)

    print(f"File: {file}, {height_node.type}")

    height_data = height_node.numpy(index)
    normal_data = normal_node.full().numpy(index)

    # Compute Shading
    relief = relief_shade(height_data, normal_data)
    plt.imshow(relief, cmap='gray')
    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
#  data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  data = "out_ebensee.tiff"
  main(data)