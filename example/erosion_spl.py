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

  for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])

  plt.tight_layout()
  plt.show()

def main():

  res = np.array([1024, 1024])          # Resolution [px]
  ext = np.array([8192.0, 8192.0])      # World Extent [m] (x, y)
  scale_z = 1.0                        # Value Scale [m/unit]

  shape = silt.shape(*res)
  pscale = [ext[0] / res[0], ext[1] / res[1], scale_z]

  # Heigth and Discharge Fields
  H = silt.tensor.from_numpy(np.zeros(tuple(res), dtype=np.float32)).to_gpu()
  Q = silt.tensor.from_numpy(np.zeros(tuple(res), dtype=np.float32)).to_gpu()
  rate = silt.like(H)

  # Input Fields (Uplift, Rainfall)
  U = soil.util.load_png(f"{DATA_DIR}/uplift.png", host=silt.gpu)
  P = soil.util.load_png(f"{DATA_DIR}/rainfall.png", host=silt.gpu)

  # Simulate

  timer = soil.timer()
  steps = 128
  for i in range(steps):
    with timer:

      graph = soil.steepest(H, soil.d8)
      Q = soil.accumulate(graph, P, soil.d8)
      S = soil.slope(H, graph, [pscale[0], pscale[1], 1.0])
      soil.spl_rate_(rate, Q, S, K=1.0, m=0.4, n=1.0)
      silt.add_(H, U)
      soil.spl_apply_(H, rate, dt=1.0)

      print(f"Step {i}/{steps}, Execution Time: {timer.count} ms")

  plot(H, Q, pscale)

if __name__ == "__main__":
  main()

