#!/usr/bin/env python

import os
import soillib as soil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from PIL import Image

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

  seed = 4
  buffer = soil.noise(index, seed)
  buffer = soil.bake(buffer, index)
  soil.multiply(buffer, 20.0)
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

  param = soil.param_t()
  param.lrate = 0.05
  param.momentumTransfer = 4.0
  param.gravity = 1.0
  param.depositionRate = 0.05
  param.entrainment = 0.25

  timer = soil.timer()
  with timer:
    soil.gpu_erode(model, param, 4096)
  print(f"Execution Time: {timer.count}ms")

  dataset = []
  for i in range(8):
    print(i)
    discharge.gpu()
    model.discharge = discharge
    soil.gpu_erode(model, param, 64)
    discharge.cpu()
    dataset = [*dataset, discharge.numpy(index)]

  '''
  We have computed the full dataset...
  Now let's make the image directly...
  '''

  color = "viridis"

  # Threshhold Clamp
  for data in dataset:
    data[data > 1.0] = 1.0

  image = np.full((index[0], index[1], 3), 0.0)
  image[:] = mpl.colormaps[color]((0))[:3]

  #for k, data in reversed(list(enumerate(dataset))):
  for k, data in enumerate(dataset):

    val = image
    rgb = mpl.colormaps[color](((k+1)/8))[:3]
    image = (1.0 - data[..., np.newaxis]) * image + data[..., np.newaxis] * rgb
#    image[data > 0.0] = rgb[:3]


  for k, data in enumerate(dataset):
    im = Image.fromarray(data)
    im.save(f"meander_out/data_{k}.tiff")
#    plt.imshow(np.log(1.0 + data))
#    plt.show()


  plt.imshow(image, interpolation='bilinear')
  plt.show()


  #discharge = np.log(1.0 + discharge)

  #tiff_out = soil.tiff(discharge, index)
  #tiff_out.write("/home/nickmcdonald/Datasets/discharge_test`.tiff")

  '''
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



  
  #plt.imshow(discharge.numpy(index)[4:-2, 4:-2])
  plt.show()
  '''

if __name__ == "__main__":
  main()