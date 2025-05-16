#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from __common__ import show_height, show_relief, show_discharge, show_layers, zip_save

'''
Multi Resolution GPU Erosion Example

This example simulates erosion at increasing resolutions.
The physical scale associated with the model corrects the
parameters so that the simulation occurs correctly at scale.
This can significantly reduce erosion simulation time.

Multi-Scale Erosion Procedure:
1. Construct Model at Resolution
2. Erode at Resolution
3. Construct New Model at Increased Resolution
4. Goto 2
'''

def main():

  simres = np.array([256, 256])         # Resolution [px]
  wscale = np.array([20.0, 20.0, 4.0])  # World Scale [km] (x, y, z)
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
  sediment[:] = 0.0
  
  # Construct Model

  model = soil.map_t(index, pscale)
  model.height = height.gpu()
  model.sediment = sediment.gpu()

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
  param.samples = 8192  # Number of Samples
  param.maxage = 512    # Maximum Particle Age
  param.lrate = 0.1     # Filter Learning Rate
  param.timeStep = 10.0 # 

  param.rainfall = 1.0      # Rainfall Rate [m/y]
  param.evapRate = 0.0001   # Evaporation Rate [1/s]

  param.gravity = 9.81      # Specific Gravity [m/s^2]
  param.viscosity = 0.125   # Kinematic Viscosity [m^2/s]
  param.bedShear = 0.125    # River Bed Shear [m^2/s]

  param.critSlope = 0.57      # Critical Slope [m/m]
  param.settleRate = 0.1      # Debris Settling Rate
  param.thermalRate = 0.025   # Thermal Erosion Rate
  param.debrisShear = 0.9

  param.depositionRate = 0.1  # Fluvial Deposition Rate
  param.suspensionRate = 0.5  # Fluvial Suspension Rate
  param.exitSlope = 0.025     # Boundary Slope [m/m]

  timer = soil.timer()

  def scaleup(model, data, track, oldres, simres):

    # Upscale Individual Buffers

    index = soil.index(simres)
    pscale = [wscale[0]/simres[0],
              wscale[1]/simres[1],
              wscale[2]]

    height = soil.buffer(soil.float32, index.elem(), soil.gpu)
    soil.resize(height, model.height, simres, oldres)

    sediment = soil.buffer(soil.float32, index.elem(), soil.gpu)
    soil.resize(sediment, model.sediment, simres, oldres)

    model = soil.map_t(index, pscale)
    model.height = height
    model.sediment = sediment

    # Update Tracking

    newdata = soil.data_t(index.elem())
    newtrack = soil.data_t(index.elem())

    soil.resize(newdata.mass, data.mass, simres, oldres)
    soil.resize(newdata.discharge, data.discharge, simres, oldres)
    soil.resize(newdata.momentum, data.momentum, simres, oldres)
    soil.resize(newdata.debris_momentum, data.debris_momentum, simres, oldres)
    soil.resize(newdata.debris, data.debris, simres, oldres)
    
    return model, newtrack, newdata, index, simres, pscale

  ksteps = [
    ([256, 256], 128),
    ([512, 512], 64),
    ([1024, 1024], 64),
    ([2048, 2048], 64)
  ]

  # Note: The first scale-up procedure here is redundant and can be removed.

  for nextres, steps in ksteps:
  
    model, data, track, index, simres, pscale = scaleup(model, data, track, simres, nextres)

    print(f"Simulating Resolution: {simres}")
    for i in range(steps):
      with timer:
        soil.erode(model, data, track, param, 1)
      print(f"Execution Time: {timer.count}ms")

  # Save Geotiff Output
  # Geotiff so that pixel and value scale are respected,
  # and we must also add the height of all layers.

  zip_save('/home/nickmcdonald/Datasets/erosion_multi.zip', {
    "height": model.height,
    "sediment": model.sediment,
    "discharge": data.discharge
  }, index, pscale)

if __name__ == "__main__":
  main()