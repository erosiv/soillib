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
- Some Code Interface Cleanup
- Test Application in CUDA Code
- Better Point Samplers
- Point Placement Optimization?
- Multi-Scale RBF?

In principle, we could say that a small number
of radial basis functions is always applied to
solve the broad shape. We can even use multiple
kdtrees to accelerate the multi-layer lookup,
or simply ignore that for certain layers.
    
This could dramatically improve the fit overall,
especially with position optimization.
'''

def plot_images(images):

  K = len(images)
  fig, ax = plt.subplots(1, K, figsize=(8, 4))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)
  for k, img in enumerate(images):

    im = ax[k].imshow(img, zorder=2,
      cmap='CMRmap',
      interpolation='bilinear')

  plt.show()

def main(input):

  for file, path in iter_tiff(input):

    print("Loading Digital Elevation Model...")

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu()

    print("Sampling Digital Elevation Model...")

    # Note: Replace with e.g. Halton Sampler

    K = 4096
    N = 8 * K

    center = soil.sampleN(index, K)
    sample = soil.sampleN(index, N)
    values = soil.sample_lerp(buffer, index, sample)
    #normal = soil.sample_grad(buffer, index, sample)

    print("Initializing Radial Basis Function Interpolator...")

    rbf = soil.rbf()
    rbf.init(center)
    rbf.shape = 14
    rbf.P = 6

    print("Solving Least Squares Problem...")

    matrix = rbf.matrix(sample)
    vector = rbf.vector(values)

    tmatrix = matrix.torch(soil.index([N+rbf.P, K+rbf.P]))
    tvector = vector.torch(soil.index([N+rbf.P]))

    w = torch.linalg.lstsq(tmatrix, tvector).solution

    print(torch.mean(torch.abs(w)))

    # Note: Replace with a soil from_torch method!
    rbf.set_w(soil.buffer.from_numpy(w.cpu().numpy()).gpu())

    print("Computing Estimation Error...")

    kdtree = soil.kdtree(center)
    value_est = rbf.sample(sample, kdtree)
    value_est = value_est.cpu().numpy(soil.index([N]))
    value_tru = values.cpu().numpy(soil.index([N]))
    abs_err = (value_est - value_tru)
    print("MSE:", np.sum(abs_err**2)/N)

    print("Re-Sampling Radial Basis Function Interpolator...")

    img = rbf.sample(index)

    plot_images([
      buffer.cpu().numpy(index),
      img.cpu().numpy(index),
    ])

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  #data = "/home/nickmcdonald/Datasets/erosion_large.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)