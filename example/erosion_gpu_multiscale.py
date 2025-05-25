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

  simres = np.array([128, 128])         # Resolution [px]
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
  soil.set(model.rainfall, 1.0)

  uplift = soil.noise(index, noise_param)
  soil.clamp(uplift, 0.0, 1.0)
  model.uplift = uplift.gpu()

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
  param.samples = 32768  # Number of Patricle Samples
  param.maxage = 512    # Maximum Particle Lifetime
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

    rainfall = soil.buffer(soil.float32, index.elem(), soil.gpu)
    soil.resize(rainfall, model.rainfall, simres, oldres)

    uplift = soil.buffer(soil.float32, index.elem(), soil.gpu)
    soil.resize(uplift, model.uplift, simres, oldres)

    model = soil.map_t(index, pscale)
    model.height = height
    model.sediment = sediment
    model.rainfall = rainfall
    model.uplift = uplift

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
    ([128, 128], 2048),
    ([256, 256], 4),
#    ([512, 512], 512),
    ([1000, 1000], 4),
#    ([2048, 2048], 512),
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

  zip_save('/home/nickmcdonald/Datasets/erosion_multi_base.zip', {
    "height": model.height,
    "sediment": model.sediment,
    "discharge": data.discharge
  }, index, pscale)

if __name__ == "__main__":
  main()