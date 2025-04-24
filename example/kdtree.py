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

  '''
  # def plot_quiver()
#   q = g.quiver()

#  acc = g.acc().cpu().numpy().transpose()
  '''

def plot_pcl_3D(points, colors = None, normals = None):

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  N = points.elem

  points = points.cpu().numpy(soil.index([N]))
  xs = points[:, 0]
  ys = points[:, 1]
  zs = points[:, 2]

#    print()

  '''
  if not colors is None:
    col = colors.cpu().numpy(soil.index([N]))
    print(np.max(col))
    col = np.log(1.0 + col)
    ax.scatter(xs, ys, zs, marker='o', c = col)
  else:
    ax.scatter(xs, ys, zs, marker='o')
  '''

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

def main(input):

  for file, path in iter_tiff(input):

    print("Loading Digital Elevation Model...")

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu()

    # Sample Random Positions in 2D,
    # lerp the height-map to get the corresponding height-values
    # concatenate these into a point-cloud map!

    K = 4096
    N = 4 * K

    # note: we could uniformly sample
    #   somehow... I guess we should try different
    #   sampling methods as well? and make sure the
    #   fit works well...

    center = soil.sampleN(index, K)
    sample = soil.sampleN(index, N)

    value = soil.sample_lerp(buffer, index, sample)

    print("Constructing Radial Basis Function Interpolator...")

    rbf = soil.rbf()
    rbf.shape = 16

    rbf.init(center)
    matrix = rbf.matrix(sample)

    tmatrix = matrix.torch(soil.index([N, K]))
    tvalue = value.torch(soil.index([N]))

    print("Solving Least Squares...")
    w = torch.linalg.lstsq(tmatrix, tvalue).solution
    rbf.set_w(soil.buffer.from_numpy(w.cpu().numpy()).gpu())

    print("Weights Set!")

    kdtree = soil.kdtree(center)
    img = rbf.sample(kdtree, index)

    plot_images([
      buffer.cpu().numpy(index),
      img.cpu().numpy(index),
    ])

#    pps = center.cpu().numpy(soil.index([K]))
#    plt.scatter(pps[:, 1], pps[:, 0], marker='x', color="black")
    
    '''
    normal = soil.sample_grad(buffer, index, pos)
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