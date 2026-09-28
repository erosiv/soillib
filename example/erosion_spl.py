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
  color,
  scale
):

  # height = height.to_cpu().numpy()

  normal = soil.normal(height, scale).copy_to(silt.cpu).numpy()
  normal_rgb = np.clip(0.5 + 0.5 * normal, 0.0, 1.0)

  discharge = discharge.to_cpu().numpy()

  fig, ax = plt.subplots(2, figsize=(12, 12))
  # silt.divide_(color, discharge)

  ax[0].imshow(normal_rgb)
  ax[0].set_title("Surface Normal")

#  ax[1].imshow(color)
#  ax[1].set_title("Surface Color")

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
  scale_z = 50.0                        # Value Scale [m/unit]
  dt = 4.0

  shape = silt.shape(*res)
  scale = [ext[0] / res[0], ext[1] / res[1], scale_z]

  # Heigth and Discharge Fields
  H = silt.tensor.from_numpy(np.zeros(tuple(res), dtype=np.float32)).to_gpu()
  Qw = silt.tensor.from_numpy(np.zeros(tuple(res), dtype=np.float32)).to_gpu()
  Qs = silt.tensor.from_numpy(np.zeros(tuple(res), dtype=np.float32)).to_gpu()
  rate = silt.like(H)

  # Input Fields (Uplift, Rainfall)
  P = soil.util.load_png(f"{DATA_DIR}/rainfall.png", host=silt.gpu)
  C = soil.util.load_png(f"{DATA_DIR}/albedo.png", host=silt.gpu)
  U = soil.util.load_png(f"{DATA_DIR}/uplift.png", host=silt.gpu)
  Uc = soil.cone(shape, apex=[0,0], height=0.0, dUdx=1e-4, dUdy=2e-4, scale=[scale[0], scale[1]])
  silt.add_(U, Uc)
  silt.multiply_(U, 1.0)

  rng = silt.rand(shape, 0)
  relax = silt.tensor.from_numpy(np.zeros(tuple(res), dtype=np.float32)).to_gpu()

  # Simulate

  timer = soil.timer()
  steps = 512
  for i in range(steps):
    with timer:

      # Gradient and Slope
      #grad = soil.grad_steepest(H, scale)
      grad = soil.grad_quadratic(H, scale)
      #grad = soil.grad_centered(H, scale)
      S = soil.slope(grad)

      # SPL Rate
      E = soil.spl_rate(Qw, S, K=0.1, m=0.4, n=1.0)
      D = soil.spl_deposition(Qs, Qw, k_S=0.01)

      N = silt.add(E, silt.multiply(D, -1.0))   # net erosion = E - D
      silt.add_(N, silt.multiply(U, -1.0))

      # soil.spl_apply_(H, N, dt = dt / scale_z)  # height -= dt * (E - D)
      soil.spl_relax_(relax, Qw, Qs, K=0.1, m=0.4, n=1.0, scale=scale)
      soil.spl_apply_exp_(H, N, relax, dt = dt / scale_z)

      # spill_limit(height, before, epsilon=1e-6) -> tensor

      # Transport Simulation

      
      # One Sample:
#      G = soil.steepest(H, soil.d8)
#      G = soil.random_weighted(H, soil.d8, rng)
#      Qw = soil.accumulate(G, P, soil.d8) # Water Discharge
#      Qs = soil.accumulate(G, E, soil.d8) # Sediment Load
#      Qc = soil.accumulate(G, C, soil.d8) # Color Discharge

      # Temporal Delay
      # alpha = 0.5
      # silt.mix_(Qw, soil.accumulate(G, P, soil.d8), alpha)
      # silt.mix_(Qs, soil.accumulate(G, E, soil.d8), alpha)
      # silt.mix_(Qc, soil.accumulate(G, C, soil.d8), alpha)

      # MultiSample:

      samples = 4

      # First draw allocates, the rest accumulate into it.
      G = soil.random_weighted(H, soil.d8, rng)
      Qw = soil.accumulate(G, P, soil.d8)   # Water Discharge
      Qs = soil.accumulate(G, E, soil.d8)   # Sediment Load
      # Qc = soil.accumulate(G, C, soil.d8)   # Color Discharge

      for _ in range(samples - 1):
        G = soil.random_weighted(H, soil.d8, rng)
        silt.add_(Qw, soil.accumulate(G, P, soil.d8))
        silt.add_(Qs, soil.accumulate(G, E, soil.d8))
        # silt.add_(Qc, soil.accumulate(G, C, soil.d8))

      silt.multiply_(Qw, 1.0 / samples)
      silt.multiply_(Qs, 1.0 / samples)
      # silt.multiply_(Qc, 1.0 / samples)

      # Diffusion Component onto Advective Component
#      diff = soil.spl_diffusivity(Qw, S, D0=0.25, k_D=0.25, m=-0.4, n=-1.0)
#      diff = silt.clamp(diff, 0.0, 8.0)
#      Qw = soil.gaussian_blur(Qw, diff)
#      Qs = soil.gaussian_blur(Qs, diff)
#      Qc = soil.gaussian_blur(Qc, diff)

      print(f"Step {i}/{steps}, Execution Time: {timer.count} ms")

  plot(H, Qw, C, scale)

if __name__ == "__main__":
  main()