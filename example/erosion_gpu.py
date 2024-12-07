#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def relief_shade(h, n):

  # Regularize Height
  h_min = np.nanmin(h)
  h_max = np.nanmax(h)
  h = (h - h_min)/(h_max - h_min)

  # Light Direction, Diffuse Lighting
  light = np.array([-1, 1, 2])
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

def main():

  index = soil.index([512, 512])

  seed = 1
  buffer = soil.noise(index, seed)
  buffer = soil.bake(buffer, index)
  soil.multiply(buffer, 80.0)
  buffer.gpu()

  discharge = soil.buffer(soil.float32, index.elem())
  soil.set(discharge, 0.0)
  discharge.gpu()

  momentum = soil.buffer(soil.vec2, index.elem())
  soil.set(momentum, [0.0, 0.0])
  momentum.gpu()

  model = soil.model_t(index)
  model.height = buffer
  model.discharge = discharge
  model.momentum = momentum

  with soil.timer() as timer:
    soil.gpu_erode(model, 512, 512)

  buffer.cpu()
  discharge.cpu()

  tiff_out = soil.tiff(buffer, index)
  tiff_out.write("/home/nickmcdonald/Datasets/erosion_gpu.tiff")

#  normal = soil.normal(buffer, index)
  height = buffer.numpy(index)
  print(np.max(height))
  print(np.min(height))

#
#  plt.imshow(height)
#  plt.show()
#
#  normal = normal.numpy(index)
#
#  relief = relief_shade(height, normal)
#


#  plt.imshow(relief, cmap='gray')



  plt.imshow(np.log(1.0 + discharge.numpy(index)))
  #plt.imshow(discharge.numpy(index)[4:-2, 4:-2])
  plt.show()

if __name__ == "__main__":
  main()