#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from __common__ import relief_shade

def main():

  index = soil.index([512, 512])

  seed = 0
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

  suspended = soil.buffer(soil.float32, index.elem())
  soil.set(suspended, 0.0)
  suspended.gpu()

  model = soil.model_t(index)
  model.height = buffer
  model.discharge = discharge
  model.momentum = momentum
  model.suspended = suspended

  param = soil.param_t()
  param.gravity = 1
  param.momentumTransfer = 0.25
  param.maxdiff = 0.75
  param.settling = 0.75
  param.depositionRate = 0.25
  param.entrainment = 0.125
  param.lrate = 0.125
  param.exitSlope = 0.0
  param.maxage = 1024
  param.hscale = 0.01
  param.evapRate = 0.001

  timer = soil.timer()
  for i in range(16):
    with timer:
      soil.gpu_erode(model, param, 4, 8192)
    print(f"Execution Time: {timer.count}ms")

  buffer.cpu()
  discharge.cpu()

  tiff_out = soil.tiff(buffer, index)
  tiff_out.write("/home/nickmcdonald/Datasets/erosion_gpu.tiff")

  normal = soil.normal(buffer, index)
  height = model.height
  height.cpu()
  height = height.numpy(index)
  print(np.max(height))
  print(np.min(height))

#  plt.imshow(height)
#  plt.show()
#
#  return
#
  normal = normal.numpy(index)
  relief = relief_shade(height, normal)
  plt.imshow(relief, cmap='gray')



#  plt.imshow(np.log(1.0 + discharge.numpy(index)))
  #plt.imshow(discharge.numpy(index)[4:-2, 4:-2])
  plt.show()

if __name__ == "__main__":
  main()