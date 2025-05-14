#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np
from tqdm import tqdm
import torch

'''
radial basis functions (with kdtree)

This is an improved version of radial basis
function fitting with added polynomial support.

The computation can be accelerated with a kdtree.
Currently, the fitting procedure occurs in pytorch
using a linear algebra least-squares solver, which
could potentially be moved into C++ for uniformity.

Todo:
- Test Application in CUDA Code
- Multi-Scale RBF?

In principle, we could say that a small number
of radial basis functions is always applied to
solve the broad shape. We can even use multiple
kdtrees to accelerate the multi-layer lookup,
or simply ignore that for certain layers.
    
This could dramatically improve the fit overall,
especially with position optimization.
'''

def main(input):

  print("Sampling Digital Elevation Model...")

  # Note: Replace with e.g. Halton Sampler

  K = 4096
  index = soil.index([512, 512])
  center = soil.sample_halton(index, K)

  print("Initializing Radial Basis Function Interpolator...")

  rbf = soil.rbf()
  rbf.init(center)
  rbf.shape = 1.25 * np.sqrt(index.elem() / K)
  rbf.P = 0

  w = torch.randn((K + rbf.P,))
  w = w.to(device='cuda')

  rbf.set_w(soil.buffer.from_torch(w))
  img = rbf.sample(index)

  '''
  Erode RBF
  '''

  simres = np.array([512, 512])         # Resolution [px]
  wscale = np.array([20.0, 20.0, 4.0])  # World Scale [km] (x, y, z)
  nscale = np.array([20.0, 20.0])       # Noise Feature Scale [km] (x, y)
  pscale = [rbf.shape,                  # Pixel Scale [km/px]
            rbf.shape,
            wscale[2]]                  # Value Scale [km/unit]

  model = soil.map_rbf(rbf, index, pscale)
  data = soil.data_t(K)
  track = soil.data_t(K)

  data.discharge[:] = 0.0
  data.momentum[:] = [0.0, 0.0]
  data.mass[:] = 0.0
  data.debris[:] = 0.0
  data.debris_momentum[:] = [0.0, 0.0]

  param = soil.param_t()
  param.samples = 8192  # Number of Samples
  param.maxage = 512     # Maximum Particle Age
  param.lrate = 0.1     # Filter Learning Rate
  param.timeStep = 10.0 # 

  param.rainfall = 1.0      # Rainfall Rate [m/y]
  param.evapRate = 0.0001   # Evaporation Rate [1/s]

  param.gravity = 9.81      # Specific Gravity [m/s^2]
  param.viscosity = 0.125   # Kinematic Viscosity [m^2/s]
  param.bedShear = 0.125    # River Bed Shear [m^2/s]

  param.critSlope = 0.57      # Critical Slope [m/m]
  param.settleRate = 0.1      # Debris Settling Rate
  param.thermalRate = 0.0   # Thermal Erosion Rate
  param.debrisShear = 0.9

  param.depositionRate = 0.1  # Fluvial Deposition Rate
  param.suspensionRate = 1.0  # Fluvial Suspension Rate
  param.exitSlope = 0.025     # Boundary Slope [m/m]

  timer = soil.timer()
  for i in range(64):
    with timer:
      soil.erode_rbf(model, data, track, param, 1)
    print(f"Execution Time: {timer.count}ms")

  '''
  Display RBF
  '''

  eroded = model.rbf.sample(index)

  plot_images([
    img.cpu().numpy(index),
    eroded.cpu().numpy(index),
  ])

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_large.tiff"
#  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)