#!/usr/bin/env python

"""
GPU Erosion Example

Single-resolution stochastic geotransport erosion: seeds a height field
from Perlin/simplex noise, then runs soillib's fluvial + debris-flow
transport kernels forward through soil.erosion.ErosionModel, which is the
one place the kernel choreography (transport_fluvial / transport_debris /
mass_transfer / layer_merge) needs to be spelled out -- see
soil/erosion.py.

Parameter values below are erosiv's own reference values for this model
(from the working erosion node this example was ported from), not fitted
to this particular resolution/scale.
"""

import numpy as np
import matplotlib.pyplot as plt

import silt
import soillib as soil


def noise_height(shape, world_scale, feature_scale, seed=3):
  """A Perlin/simplex height field to erode, in world_scale's z-units."""
  param = soil.noise_t()
  param.ext = np.array([shape[0], shape[1]]) * (feature_scale / world_scale[0:2])
  param.seed = seed
  return soil.noise(shape, param)


def main():

  simres = np.array([256, 256])         # Resolution [px]
  wscale = np.array([20.0, 20.0, 4.0])  # World Scale [km] (x, y, z)
  nscale = np.array([20.0, 20.0])       # Noise Feature Scale [km] (x, y)
  pscale = [wscale[0] / simres[0],      # Pixel Scale [km/px]
            wscale[1] / simres[1],
            wscale[2]]                  # Value Scale [km/unit]

  shape = silt.shape(*simres)

  # Reference Parameterization
  # (erosiv's own working values for this erosion model)

  param = soil.param_t()

  param.maxage = 2048       # Maximum Particle Lifetime
  param.lrate = 1.0         # Filter Learning Rate
  param.timeStep = 250.0    # Geological Timestep [y]

  param.exitSlope = 0.01    # Boundary Slope Condition
  param.uplift = 0.001      # Uplift Rate [m/y]
  param.rainfall = 1.0      # Rainfall Rate [m/y]
  param.gravity = 9.81      # Specific Gravity [m/s^2]
  param.evapRate = 0.0      # Evapotranspiration Rate [1/s]

  param.frictionFactor = 0.06
  param.fluvialExponent = 2.0

  param.suspensionRateFluvial = 0.0008 * (0.0075 ** 2.0)
  param.depositionRateFluvial = 0.4
  param.suspensionRateDebris = 0.001
  param.depositionRateDebris = 0.001
  param.landslideRateDebris = 0.01

  param.critSlopeBedrock = 0.57
  param.critSlopeSediment = 0.3
  param.yieldStress = 0.001

  param.viscosityWater = 0.000001
  param.bedShearWater = 0.0075
  param.densityWater = 1.0

  param.viscosityDebris = 0.0
  param.bedShearDebris = 0.99
  param.densityDebris = 2.0

  # Model Setup

  model = soil.erosion.ErosionModel(shape, pscale, param)
  model.set_height(noise_height(shape, wscale, nscale))

  # A uniform uplift field, so param.uplift actually takes effect and the
  # terrain settles towards an uplift/erosion equilibrium rather than just
  # smoothing away -- set to 0 for a pure "erode this noise" demo instead.
  model.set_uplift(np.ones(tuple(simres), dtype=np.float32))

  # Simulate

  timer = soil.timer()
  steps = 512
  for i in range(steps):
    with timer:
      model.step(1)
    if i % 32 == 0:
      print(f"Step {i}/{steps}, Execution Time: {timer.count} ms")

  # Display

  soil.erosion.plot(model)
  soil.util.show_discharge(model.discharge)


if __name__ == "__main__":
  main()
