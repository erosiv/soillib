#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def main(path):

  path = os.fsencode(path)
  
  if not os.path.exists(path):
    raise RuntimeError("path does not exist")
  
  def relief_shade(h, n):

    # Regularize Height
    h_min = np.min(h)
    h_max = np.max(h)
    h = (h - h_min)/(h_max - h_min)

    # Light Direction, Diffuse Lighting
    light = np.array([ 1,-1, 2])
    light = light / np.linalg.norm(light)

    diffuse = np.sum(light * n, axis=-1)
    diffuse = 0.05 + 0.9*diffuse

    # Flat-Toning
    flattone = np.full(h.shape, 0.9)
    weight = 1.0 - n[:,:,2]
    weight = weight * (1.0 - h * h)

    # Full Diffuse Shading Value
    diffuse = (1.0 - weight) * diffuse + weight * flattone
    return diffuse

  def show_tiff(path):

    img = soil.geotiff(path)

    # Get the Buffer Type
    height = img.array()
    normal = soil.normal(height)

    # Extract Data
    normal_data = np.array(normal())
    height_data = np.array(height)
    # data = np.array(data)

    relief = relief_shade(height_data, normal_data)
    print(relief.shape, relief.dtype)

    plt.imshow(relief, cmap='gray')
    plt.show()

  if os.path.isfile(path):

    show_tiff(path)

  elif os.path.isdir(path):
    for file in os.listdir(path):
      show_tiff(os.path.join(path, file))

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  main(data)