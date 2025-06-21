#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from __common__ import show_discharge, show_height

def main():

  simres = np.array([1000, 1000])         # Resolution [px]
  wscale = np.array([20.0, 20.0, 4.0])  # World Scale [km] (x, y, z)
  nscale = np.array([20.0, 20.0])       # Noise Feature Scale [km] (x, y)
  pscale = [wscale[0]/simres[0],        # Pixel Scale [km/px]
            wscale[1]/simres[1],
            wscale[2]]                  # Value Scale [km/unit]

  noise_param = soil.noise_t()
  noise_param.ext = simres * nscale / wscale[0:2]
  noise_param.seed = 3

  index = soil.index(simres)  
  height = soil.noise(index, noise_param)
  soil.multiply(height, 1.0)

  sediment = soil.buffer(soil.float32, index.elem(), soil.gpu)
  sediment[:] = 0.0

  # Construct Model

  model = soil.map_t(index, pscale)
  model.height = height.gpu()
  model.sediment = sediment.gpu()

  model.rainfall = soil.buffer(soil.float32, index.elem(), soil.gpu)
  model.rainfall[:] = 1.0

  model.uplift = soil.buffer(soil.float32, index.elem(), soil.gpu)
  model.uplift[:] = 0.0

  # Construct Data

  data = soil.data_t(index.elem())
  track = soil.data_t(index.elem())

  data.discharge[:] = 0.0
  data.momentum[:] = [0.0, 0.0]
  data.mass[:] = 0.0
  data.debris[:] = 0.0
  data.debris_momentum[:] = [0.0, 0.0]

  # Construct Parameters

  param = soil.param_t()
  param.timeStep = 10.0 # Geological Timestep [y]
  param.samples = 8192  # Number of Patricle Samples
  param.maxage = 256    # Maximum Particle Lifetime
  param.lrate = 1.0     # Filter Learning Rate

  param.gravity = 9.81    # Specific Gravity [m/s^2]
  param.uplift = 0.01     # Uplift Rate [m/y]
  
  param.rainfall = 1.0              # Rainfall Rate [m/y]
  param.evapRate = 0.0005           # Evapotranspiration Rate [1/s]
  param.viscosity = 0.000001        # Water Viscosity [m^2/s]
  param.bedShear = 12.5             # Turbulent Shear Stress [Pa = kg/m/s^2]
  param.suspensionRate = 0.0000008  # Fluvial Suspension Rate
  param.depositionRate = 0.00001    # Fluvial Deposition Rate
  param.fluvialExponent = 0.01      # Fluvial Power Exponent
  param.exitSlope = 0.025           # Slope Boundary Condition

  param.critSlope = 0.57                # Critical Slope
  param.debrisCreepRate = 0.0025        # Landslide Erosion Rate
  param.debrisSuspensionRate = 0.00025  # Debris Suspension Rate
  param.debrisDepositionRate = 0.0001   # Debris Deposition Rate
  param.debrisYieldStress = 2E6         # Yield Stress [Pa]
  param.debrisDensity = 2500.0          # Debris Density [kg/m^3]
  param.debrisViscosity = 0.004         # Debris Viscosity [m^2/s]
  param.debrisBedShear = 60/2500.0      # Debris Turbulent Shear Stress

  timer = soil.timer()
  for i in range(2048):
    with timer:
      soil.erode(model, data, track, param, 1)
    print(f"Execution Time: {timer.count}ms")

#  tiff_out = soil.tiff(height.cpu(), index)
#  tiff_out.write("/home/nickmcdonald/Datasets/erosion_gpu.tiff")

#  show_discharge(model.discharge, index)
  show_height(model.height.cpu(), index)

if __name__ == "__main__":
  main()