#!/usr/bin/env python

"""
GPU Erosion Example -- Publication Reproduction

Stochastic geotransport erosion, parameterized to reproduce the title
scene of "Stochastic Geomorphological Transport for Terrain Erosion
Simulation" (https://erosiv.studio/publications/stochastic-geomorphological-transport).

Domain, input maps and erosion parameters are taken verbatim from the
scene's original parameterization, so this headless script reproduces the
same result given the same input maps.

Uses soil.erosion.ErosionModel for the kernel choreography
(transport_fluvial / transport_debris / mass_transfer / layer_merge) --
see soil/erosion.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import silt
import soillib as soil

DATA_DIR = "data/scene1"

def plot(
  height,
  discharge,
  scale
):

  # height = height.to_cpu().numpy()

  normal = soil.normal(height, scale).copy_to(silt.cpu).numpy()
  normal_rgb = np.clip(0.5 + 0.5 * normal, 0.0, 1.0)

  discharge = discharge.to_cpu().numpy()

  fig, ax = plt.subplots(2, figsize=(12, 12))

  ax[0].imshow(normal_rgb)
  ax[0].set_title("Surface Normal")

  ax[1].imshow(1.0 + discharge, cmap='CMRmap',
                   norm=colors.LogNorm(1, 1.0 + np.nanmax(discharge)))
  ax[1].set_title("Discharge")

#  ax[0, 1].imshow(albedo)
#  ax[0, 1].set_title("Surface Albedo")
#
#  ax[1, 0].imshow(shaded_albedo)
#  ax[1, 0].set_title("Relief-Shaded Albedo")

  for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])

  plt.tight_layout()
  plt.show()

def main():

  res = np.array([1024, 1024])          # Resolution [px]
  ext = np.array([8192.0, 8192.0])      # World Extent [m] (x, y)
  scale_z = 50.0                        # Value Scale [m/unit]
  
  shape = silt.shape(*res)
  scale = [
    ext[0] / res[0],
    ext[1] / res[1],
  ]

  height = silt.tensor.from_numpy(np.zeros(tuple(res), dtype=np.float32)).to_gpu()
  uplift = soil.util.load_png(f"{DATA_DIR}/uplift.png", host=silt.gpu)
  # uplift = soil.gaussian_blur(uplift, 2.0)
  rainfall = soil.util.load_png(f"{DATA_DIR}/rainfall.png", host=silt.gpu)
  discharge = silt.tensor.from_numpy(np.zeros(tuple(res), dtype=np.float32)).to_gpu()

  # Simulate

  timer = soil.timer()
  steps = 128
  for i in range(steps):
    with timer:

      silt.add_(height, uplift)
      graph = soil.steepest(height, soil.d8)
      discharge = soil.accumulate(graph, rainfall, soil.d8)
      S = soil.slope(height, graph, scale)
      rate = soil.spl_rate(discharge, S, 1.0, 0.4, 1.0)
      soil.spl_apply(height, rate, 1.0)

      print(f"Step {i}/{steps}, Execution Time: {timer.count} ms")
  

  plot(height, discharge, [*scale, 50])

  # Export

  # written = soil.erosion.export(model, out_dir="out")
  # print(f"Exported {len(written)} files to out/")

if __name__ == "__main__":
  main()

