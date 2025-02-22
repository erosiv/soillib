#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from __common__ import relief_shade

def main():

  index = soil.index([512, 512])  # Simulation Resolution
  wscale = [40.0, 40.0, 4.0]      # World Scale [km] (x, y, z)
  nscale = [20.0, 20.0]           # Noise Feature Scale [km] (x, y)
  scale = [ wscale[0]/index[0],   # Pixel Scale [km]
            wscale[2], 
            wscale[1]/index[1]] 

  noise_param = soil.noise_t()
  noise_param.ext = np.array([index[0], index[1]]) * np.array(nscale) / np.array([wscale[0], wscale[1]])
  noise_param.seed = 0

  height = soil.noise(index, noise_param)
  soil.multiply(height, 1.0)

  sediment = soil.buffer(soil.float32, index.elem(), soil.gpu)
  discharge = soil.buffer(soil.float32, index.elem(), soil.gpu)
  momentum = soil.buffer(soil.vec2, index.elem(), soil.gpu)
  
  sediment[:] = 0.0
  discharge[:] = 0.0
  momentum[:] = [0.0, 0.0]
  
  model = soil.model_t(index, scale)
  model.height = height.gpu()
  model.sediment = sediment
  model.discharge = discharge
  model.momentum = momentum

  param = soil.param_t()
  param.samples = 8192  # Number of Path Samples
  param.maxage = 128    # Maximum Path Length
  param.lrate = 0.2     # Filter Learning Rate

  param.rainfall = 2.0      # Rainfall Rate [m/y]
  param.evapRate = 0.0001   # Evaporation Rate [1/s]

  param.gravity = 9.81      # Specific Gravity [m/s^2]
  param.viscosity = 0.03    # Kinematic Viscosity [m^2/s]

  param.maxdiff = 0.57      # Critical Slope [m/m]
  param.settling = 0.005    # Thermal Erosion Rate

  param.depositionRate = 0.05 # Fluvial Deposition Rate
  param.entrainment = 0.00025 # Fluvial Suspension Rate
  param.exitSlope = 0.01      # Boundary Slope [m/m]

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