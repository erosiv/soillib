#!/usr/bin/env python

"""
GPU Erosion Example -- Publication Reproduction

Stochastic geotransport erosion, parameterized to reproduce the title
scene of "Stochastic Geomorphological Transport for Terrain Erosion
Simulation" (https://erosiv.studio/publications/stochastic-geomorphological-transport).

This headless script node should converge to the same result given the same input maps.

Uses soil.erosion.ErosionModel for the kernel choreography
(transport_fluvial / transport_debris / mass_transfer / layer_merge) --
see soil/erosion.py.
"""

import numpy as np
import matplotlib.pyplot as plt

import silt
import soillib as soil

DATA_DIR = "data/scene1"

def main():

  # Domain

  res = np.array([1024, 1024])          # Resolution [px]
  ext = np.array([8192.0, 8192.0])      # World Extent [m] (x, y)
  scale_z = 50.0                        # Value Scale [m/unit]
  pscale = [ext[0] / res[0], ext[1] / res[1], scale_z]

  shape = silt.shape(*res)

  # Erosion Parameters

  param = soil.param_t()

  param.maxage = 2048
  param.lrate = 1.0
  param.timeStep = 250.0

  param.exitSlope = 0.0
  param.uplift = 0.002
  param.rainfall = 1.0
  param.gravity = 9.81
  param.evapRate = 0.0

  param.frictionFactor = 0.06
  param.fluvialExponent = 2.0

  param.depositionRateFluvial = 0.4
  param.suspensionRateFluvial = 4.5e-08
  param.suspensionRateDebris = 0.01
  param.depositionRateDebris = 0.5
  param.landslideRateDebris = 0.0025

  param.critSlopeBedrock = 0.57
  param.critSlopeSediment = 0.1
  param.yieldStress = 0.05

  param.viscosityWater = 1.0e-06
  param.bedShearWater = 0.0075
  param.densityWater = 1.0

  param.viscosityDebris = 0.0
  param.bedShearDebris = 0.025
  param.densityDebris = 2.0

  param.force = [0.0, 0.0]

  samples = 131072

  # Model Setup -- flat bedrock, driven entirely by uplift.png * param.uplift

  model = soil.erosion.ErosionModel(shape, pscale, param, samples=samples)
  model.set_height(np.zeros(tuple(res), dtype=np.float32))

  # Input Maps

  uplift = soil.util.load_png(f"{DATA_DIR}/uplift.png", host=silt.gpu)
  # uplift = soil.gaussian_blur(uplift, 2.0)
  model.set_uplift(uplift)

  rainfall = soil.util.load_png(f"{DATA_DIR}/rainfall.png", host=silt.gpu)
  model.set_rainfall(rainfall)

  albedo_bedrock = soil.util.load_png(f"{DATA_DIR}/albedo.png", host=silt.gpu)
  model.set_albedo_bedrock(albedo_bedrock)

  # Simulate

  timer = soil.timer()
  steps = 2048
  for i in range(steps):
    with timer:
      model.step(1)
    if i % 128 == 0:
      print(f"Step {i}/{steps}, Execution Time: {timer.count} ms")

  # Display

  soil.erosion.plot(model)
  soil.util.show_discharge(model.discharge)


if __name__ == "__main__":
  main()
