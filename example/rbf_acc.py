#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np
from tqdm import tqdm
import torch

'''
radial basis functions (with kdtree)
sparse accumulation and erosion kernels

1. Compute the RBF representation
2. Launch the sparse accumulation kernel
3. Visualize the result...
'''

def main(input):

  for file, path in iter_tiff(input):

    print("Loading Digital Elevation Model...")

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu()

    print("Sampling Digital Elevation Model...")

    K = 4096
    N = 8 * K

    center = soil.sample_halton(index, K)
    sample = soil.sampleN(index, N)
    values = soil.sample_lerp(buffer, index, sample)

    print("Initializing Radial Basis Function Interpolator...")

    rbf = soil.rbf()
    rbf.init(center)
    rbf.shape = 1.25 * np.sqrt(index.elem() / K)
    rbf.P = 6

    print("Solving Least Squares Problem...")

    matrix = rbf.matrix(sample)
    vector = rbf.vector(values)

    tmatrix = matrix.torch(soil.index([N+rbf.P, K+rbf.P]))
    tvector = vector.torch(soil.index([N+rbf.P]))

    w = torch.linalg.lstsq(tmatrix, tvector).solution
    rbf.set_w(soil.buffer.from_torch(w))

    print("Computing Sparse Accumulation")

    kdtree = soil.kdtree(center)
    acc = soil.sparseacc(rbf, kdtree, index, 16)
    acc = acc.cpu().numpy(soil.index([K]))
    print(acc)
    print(np.min(acc), np.max(acc))

    # ... how do we visualize this?
    pos = center.cpu().numpy(soil.index([K]))
    plt.scatter(pos[:,1], index[0]-1-pos[:,0], c=acc, 
    cmap='CMRmap_r', norm=colors.LogNorm(acc.min(), acc.max()))
    plt.show()

    # do something to plot this data...

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_large.tiff"
#  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)