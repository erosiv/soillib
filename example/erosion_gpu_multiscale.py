#!/usr/bin/env python

"""
Multi-Resolution GPU Erosion Example

The same scene as erosion_gpu.py (domain, input maps, erosion
parameters), but stepped forward through a sequence of increasing
resolutions via soil.erosion.ErosionModel.resized() (soil.resize under
the hood): coarse resolutions establish the large-scale drainage pattern
cheaply, and later, finer resolutions refine detail without restarting
the simulation from scratch.

The publication's title image was produced at a single fixed resolution
(see erosion_gpu.py), not with this multiscale schedule, so this script's
output is expected to differ from it -- this is a demonstration of the
resizing/multiscale machinery, not a reproduction.
"""

import numpy as np
import matplotlib.pyplot as plt

import silt
import soillib as soil

DATA_DIR = "data/scene1"


def main():

  # Domain -- same as erosion_gpu.py, at the finest resolution in the schedule

  ext = np.array([8192.0, 8192.0])      # World Extent [m] (x, y)
  scale_z = 50.0                        # Value Scale [m/unit]

  def pixel_scale(res):
    return [ext[0] / res[0], ext[1] / res[1], scale_z]

  # Erosion Parameters -- same as erosion_gpu.py

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

  # Input Maps -- loaded once at full resolution, resampled per stage below

  uplift_full = soil.util.load_png(f"{DATA_DIR}/uplift.png", host=silt.gpu)
  # uplift_full = soil.gaussian_blur(uplift_full, 2.0)
  rainfall_full = soil.util.load_png(f"{DATA_DIR}/rainfall.png", host=silt.gpu)
  albedo_full = soil.util.load_png(f"{DATA_DIR}/albedo.png", host=silt.gpu)

  # Resolution Schedule: (resolution, steps at that resolution)

  schedule = [
    ([128, 128], 32),
    ([256, 256], 32),
    ([512, 512], 32),
    ([1024, 1024], 32),
  ]

  # Seed at the first (coarsest) resolution -- flat bedrock, driven entirely
  # by uplift.png * param.uplift, same as erosion_gpu.py

  res0 = np.array(schedule[0][0])
  shape = silt.shape(*res0)

  model = soil.erosion.ErosionModel(shape, pixel_scale(res0), param, samples=samples)
  model.set_height(np.zeros(tuple(res0), dtype=np.float32))
  model.set_uplift(soil.resize(uplift_full, shape))
  model.set_rainfall(soil.resize(rainfall_full, shape))
  model.set_albedo_bedrock(soil.resize(albedo_full, shape))

  timer = soil.timer()

  for res, steps in schedule:

    res = np.array(res)
    new_shape = silt.shape(*res)
    if new_shape != model.shape:
      model = model.resized(new_shape, pixel_scale(res))
      # Re-source the fixed boundary condition maps directly from the
      # original full-resolution PNGs, rather than cascading resizes of
      # resizes. albedo_bedrock is set directly (not via
      # set_albedo_bedrock, which would also reset albedo_surface --
      # simulation-evolved state that resized() has already carried over
      # and should not be wiped on a resolution change).
      model.set_uplift(soil.resize(uplift_full, new_shape))
      model.set_rainfall(soil.resize(rainfall_full, new_shape))
      model.albedo_bedrock = soil.resize(albedo_full, new_shape)

    print(f"Simulating Resolution: {tuple(res)}")
    for i in range(steps):
      with timer:
        model.step(1)
    print(f"  {steps} steps, last step {timer.count} ms")

  # Display / Export

  soil.erosion.plot(model)

  written = soil.erosion.export(model, out_dir="out/multiscale")
  print(f"Exported {len(written)} files to out/multiscale/")


if __name__ == "__main__":
  main()
