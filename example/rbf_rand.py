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
  N = 8 * K

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

  plot_images([
    img.cpu().numpy(index),
    img.cpu().numpy(index),
  ])

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_large.tiff"
#  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)