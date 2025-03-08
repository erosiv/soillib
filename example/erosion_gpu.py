#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from __common__ import show_discharge

def main():

  '''
  Single Resolution GPU Erosion Example
  1. Define Physical Scale and Simulation Resolution
  2. Allocate Buffer and Construct Model Struct
  3. Define Physical Erosion Parameters
  4. Simulate
  '''

  simres = np.array([512, 512])         # Resolution [px]
  wscale = np.array([40.0, 40.0, 4.0])  # World Scale [km] (x, y, z)
  nscale = np.array([20.0, 20.0])       # Noise Feature Scale [km] (x, y)
  pscale = [wscale[0]/simres[0],        # Pixel Scale [km/px]
            wscale[1]/simres[1],
            wscale[2]]                  # Value Scale [km/unit]

  noise_param = soil.noise_t()
  noise_param.ext = simres * nscale / wscale[0:2]
  noise_param.seed = 0

  index = soil.index(simres)  
  height = soil.noise(index, noise_param)
  soil.multiply(height, 1.0)

  sediment = soil.buffer(soil.float32, index.elem(), soil.gpu)
  discharge = soil.buffer(soil.float32, index.elem(), soil.gpu)
  momentum = soil.buffer(soil.vec2, index.elem(), soil.gpu)
  
  sediment[:] = 0.0
  discharge[:] = 0.0
  momentum[:] = [0.0, 0.0]

  model = soil.model_t(index, pscale)
  model.height = height.gpu()
  model.sediment = sediment
  model.discharge = discharge
  model.momentum = momentum

  param = soil.param_t()
  param.samples = 8192  # Number of Samples
  param.maxage = 512    # Maximum Particle Age
  param.lrate = 0.2     # Filter Learning Rate
  param.timeStep = 10.0 # 

  param.rainfall = 1.0      # Rainfall Rate [m/y]
  param.evapRate = 0.0001   # Evaporation Rate [1/s]

  param.gravity = 9.81      # Specific Gravity [m/s^2]
  param.viscosity = 0.025   # Kinematic Viscosity [m^2/s]
  param.bedShear = 0.01     # River Bed Shear [m^2/s]

  param.critSlope = 0.57      # Critical Slope [m/m]
  param.settleRate = 0.1      # Debris Settling Rate
  param.thermalRate = 0.0025  # Thermal Erosion Rate

  param.depositionRate = 0.005    # Fluvial Deposition Rate
  param.suspensionRate = 0.00025  # Fluvial Suspension Rate
  param.exitSlope = 0.01          # Boundary Slope [m/m]

  timer = soil.timer()
  for i in range(1024):
    with timer:
      soil.erode(model, param, 1)
    print(f"Execution Time: {timer.count}ms")

#  tiff_out = soil.tiff(height.cpu(), index)
#  tiff_out.write("/home/nickmcdonald/Datasets/erosion_gpu.tiff")

  show_discharge(model.discharge, index)

if __name__ == "__main__":
  main()