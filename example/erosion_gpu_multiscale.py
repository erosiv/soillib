#!/usr/bin/env python

"""
Multi-Resolution GPU Erosion Example

Runs the same stochastic geotransport erosion model as erosion_gpu.py, but
steps it forward through a sequence of increasing resolutions using
soil.erosion.ErosionModel.resized() (soil.resize under the hood): coarse
resolutions establish the large-scale drainage pattern cheaply, and later,
finer resolutions refine detail without restarting the simulation from
scratch.
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

  wscale = np.array([20.0, 20.0, 4.0])  # World Scale [km] (x, y, z), fixed
  nscale = np.array([20.0, 20.0])       # Noise Feature Scale [km] (x, y)

  def pixel_scale(res):
    return [wscale[0] / res[0], wscale[1] / res[1], wscale[2]]

  # Reference Parameterization (erosiv's own working values for this model)

  param = soil.param_t()

  param.maxage = 2048
  param.lrate = 1.0
  param.timeStep = 250.0

  param.exitSlope = 0.01
  param.uplift = 0.001
  param.rainfall = 1.0
  param.gravity = 9.81
  param.evapRate = 0.0

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

  # Seed at the first (coarsest) resolution

  simres = np.array([128, 128])
  shape = silt.shape(*simres)

  model = soil.erosion.ErosionModel(shape, pixel_scale(simres), param)
  model.set_height(noise_height(shape, wscale, nscale))
  model.set_uplift(np.ones(tuple(simres), dtype=np.float32))

  # Resolution Schedule: (resolution, steps at that resolution)

  schedule = [
    ([128, 128], 2048),
    ([256, 256], 512),
    ([512, 512], 256),
    ([1024, 1024], 64),
  ]

  timer = soil.timer()

  for res, steps in schedule:

    res = np.array(res)
    new_shape = silt.shape(*res)
    if new_shape != model.shape:
      model = model.resized(new_shape, pixel_scale(res))

    print(f"Simulating Resolution: {tuple(res)}")
    for i in range(steps):
      with timer:
        model.step(1)
    print(f"  {steps} steps, last step {timer.count} ms")

  # Display / Save

  soil.erosion.plot(model)

  layers_np = model.layers.copy_to(silt.cpu).numpy()
  sediment = silt.tensor.from_numpy(np.ascontiguousarray(layers_np[..., 1]))

  soil.util.zip_save("data/erosion_multiscale.zip", {
    "height": model.height,
    "sediment": sediment,
    "discharge": model.discharge,
  }, model.scale)


if __name__ == "__main__":
  main()
