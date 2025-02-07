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
  height = soil.noise(index, seed)
  soil.multiply(height, 80.0)

  discharge = soil.buffer(soil.float32, index.elem(), soil.gpu)
  suspended = soil.buffer(soil.float32, index.elem(), soil.gpu)
  momentum = soil.buffer(soil.vec2, index.elem(), soil.gpu)
  
  discharge[:] = 0.0
  suspended[:] = 0.0
  momentum[:] = [0.0, 0.0]

  model = soil.model_t(index)
  model.height = height.gpu()
  model.discharge = discharge
  model.momentum = momentum
  model.suspended = suspended

  param = soil.param_t()
  param.samples = 8192
  param.gravity = 1
  param.momentumTransfer = 0.25
  param.maxdiff = 0.75
  param.settling = 0.75
  param.depositionRate = 0.1
  param.entrainment = 0.125
  param.lrate = 0.125
  param.exitSlope = 0.0
  param.maxage = 1024
  param.hscale = 0.01
  param.evapRate = 0.001

  timer = soil.timer()
  for i in range(1024):
    with timer:
      soil.erode(model, param, 1)
    print(f"Execution Time: {timer.count}ms")

#  tiff_out = soil.tiff(height.cpu(), index)
#  tiff_out.write("/home/nickmcdonald/Datasets/erosion_gpu.tiff")

#  normal = soil.normal(height, index)
#  height = model.height.cpu().numpy(index)

  plt.imshow(np.log(1.0 + model.discharge.cpu().numpy(index)))
  plt.show()

if __name__ == "__main__":
  main()