#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from __common__ import show_height, show_relief, show_discharge

def main():

  '''
  Multi Resolution GPU Erosion Example

  This example simulates erosion at increasing resolutions.
  The physical scale associated with the model corrects the
  parameters so that the simulation occurs correctly at scale.
  This can significantly reduce erosion simulation time.
  '''

  simres = np.array([256, 256])         # Resolution [px]
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
  param.samples = 8192  # Number of Path Samples
  param.maxage = 128    # Maximum Path Length
  param.lrate = 0.2     # Filter Learning Rate

  param.rainfall = 2.0      # Rainfall Rate [m/y]
  param.evapRate = 0.0001   # Evaporation Rate [1/s]

  param.gravity = 9.81      # Specific Gravity [m/s^2]
  param.viscosity = 0.03    # Kinematic Viscosity [m^2/s]

  param.critSlope = 0.57      # Critical Slope [m/m]
  param.settleRate = 0.005    # Debris Settling Rate
  param.thermalRate = 0.005   # Thermal Erosion Rate

  param.depositionRate = 0.05 # Fluvial Deposition Rate
  param.entrainment = 0.00025 # Fluvial Suspension Rate
  param.exitSlope = 0.01      # Boundary Slope [m/m]

  timer = soil.timer()

  '''
  Multi-Scale Erosion Procedure:
  1. Construct Model at Resolution
  2. Erode at Resolution
  3. Construct New Model at Increased Resolution
  4. Goto 2
  '''

  def scaleup(model, oldres, simres):

    # Upscale Individual Buffers

    index = soil.index(simres)
    pscale = [wscale[0]/simres[0],
              wscale[1]/simres[1],
              wscale[2]]

    height = soil.buffer(soil.float32, index.elem(), soil.gpu)
    sediment = soil.buffer(soil.float32, index.elem(), soil.gpu)
    discharge = soil.buffer(soil.float32, index.elem(), soil.gpu)
    momentum = soil.buffer(soil.vec2, index.elem(), soil.gpu)

    soil.resize(height, model.height, simres, oldres)
    soil.resize(sediment, model.sediment, simres, oldres)
    soil.resize(discharge, model.discharge, simres, oldres)
    soil.resize(momentum, model.momentum, simres, oldres)

    model = soil.model_t(index, pscale)
    model.height = height
    model.sediment = sediment
    model.discharge = discharge
    model.momentum = momentum

    return model, index, simres, pscale

  ksteps = [
    ([256, 256], 512),
    ([512, 512], 256),
    ([1024, 1024], 128),
#    ([2048, 2048], 128)
  ]

  # Note: The first scale-up procedure here is redundant and can be removed.

  for nextres, steps in ksteps:
  
    model, index, simres, pscale = scaleup(model, simres, nextres)

    print(f"Simulating Resolution: {simres}")
    for i in range(steps):
      with timer:
        soil.erode(model, param, 1)
      print(f"Execution Time: {timer.count}ms")

  # Save Geotiff Output
  # Geotiff so that pixel and value scale are respected,
  # and we must also add the height of all layers.

  height = model.height
  soil.add(height, model.sediment)

  tiff_out = soil.geotiff(height.cpu(), index)
  tiff_out.meta.scale = pscale  # Pixel Scale Important!
  tiff_out.write("/home/nickmcdonald/Datasets/erosion_gpu_multi.tiff")

  show_relief(height, index, pscale)
  show_discharge(model.discharge, index)

if __name__ == "__main__":
  main()