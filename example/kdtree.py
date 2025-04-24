#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np
from tqdm import tqdm
import torch

'''
kdtree kernel tests
---

Basically, right now this is not working how I want it to.
I don't want to stash the whole matrix away, given that it
is quite large, but in principle it's just a large image.

Instead, what if we just use pytorch to solve the least
squares problem directly, then use the weights directly
in the sampling?

Let's try it.
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

'''
def plot_pcl(points, colors = None, normals = None):

  fig = plt.figure()
  ax = fig.add_subplot()

  N = points.elem

  points = points.cpu().numpy(soil.index([N]))
  X = points[:, 0]
  Y = points[:, 1]
  Z = points[:, 2]

  col = colors.cpu().numpy(soil.index([N]))
  col = np.log(1.0 + col)

#  ax.plot(X, Y, 'o', markersize=2, color='grey')
  ax.tripcolor(X, Y, col, shading='gouraud')

  normals = normals.cpu().numpy(soil.index([N]))
  normals = normals[:, 0:2]
  norm = np.sqrt(np.sum(normals * normals, axis=-1))
  normals = normals / np.expand_dims(norm, axis=-1)
  U = 5.0 * normals[:, 0]
  V = 5.0 * normals[:, 1]

#  ax.quiver(X, Y, U, V, color="black", angles='xy', scale_units='xy', scale=1, width=.0015)
#    headwidth=0, headaxislength=0, headlength=0)

  plt.show()

def plot_pcl_3D(points, colors = None, normals = None):

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  N = points.elem

  points = points.cpu().numpy(soil.index([N]))
  xs = points[:, 0]
  ys = points[:, 1]
  zs = points[:, 2]

  normals = normals.cpu().numpy(soil.index([N]))
  normals = 0.5 + 0.5*normals
  ax.scatter(xs, ys, zs, marker='o', c=normals)

  zmin = np.min(zs)
  zmax = np.max(zs)
  zmid = 0.5*(zmin + zmax)

  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')
  ax.set_zlim(zmid-256, zmid+256)
  plt.show()
'''

def main(input):

  for file, path in iter_tiff(input):

    print("Loading Digital Elevation Model...")

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu()

    print("Sampling Digital Elevation Model...")

    # Note: Replace with e.g. Halton Sampler

    K = 2048
    N = 8 * K

    center = soil.sampleN(index, K)
    sample = soil.sampleN(index, N)

    value = soil.sample_lerp(buffer, index, sample)
    #normal = soil.sample_grad(buffer, index, sample)

    print("Initializing Radial Basis Function Interpolator...")

    rbf = soil.rbf()
    rbf.init(center)
    rbf.shape = 24
    rbf.P = 3

    print("Solving Least Squares Problem...")

    matrix = rbf.matrix(sample)
    tmatrix = matrix.torch(soil.index([N+rbf.P, K+rbf.P]))
    tvalue = value.torch(soil.index([N]))
    
    # note: this part could also be automated...
    if rbf.P == 3:
      pvalue = torch.tensor([0, 0, 0]).to(device='cuda')
      tvalue = torch.cat((tvalue, pvalue))
    w = torch.linalg.lstsq(tmatrix, tvalue).solution

    # Note: Replace with a soil from_torch method!
    rbf.set_w(soil.buffer.from_numpy(w.cpu().numpy()).gpu())

    print("Computing Estimation Error...")

    value_est = rbf.sample(sample)
    value_est = value_est.cpu().numpy(soil.index([N]))
    value_tru = value.cpu().numpy(soil.index([N]))
    abs_err = (value_est - value_tru)
    print("MSE:", np.sum(abs_err**2)/N)

    print("Re-Sampling Radial Basis Function Interpolator...")

    img = rbf.sample(index)

    plot_images([
      buffer.cpu().numpy(index),
      img.cpu().numpy(index),
    ])

#    pps = center.cpu().numpy(soil.index([K]))
#    plt.scatter(pps[:, 1], pps[:, 0], marker='x', color="black")
    
    '''
   
#    kdtree = soil.kdtree(center)
    rbf.fit(kdtree, pcl, 128)

    values = rbf.sample(kdtree, pos)
    pcl2 = soil.concat(pos, values)
    
    values = values.cpu().numpy(soil.index([N]))
    height = height.cpu().numpy(soil.index([N]))

    err = (values - height)
    print(np.sum(err * err)/N)

    img = 
    plt.imshow(img)
    plt.show()
    '''

    '''
    print("Computing Accumulation...")
    acc = soil.sparseacc(kdtree, pcl, normal, index, 64)
    plot_pcl_3D(pcl2, None, normal)
    return
    '''

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  #data = "/home/nickmcdonald/Datasets/erosion_large.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)