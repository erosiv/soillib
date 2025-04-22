#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np
from tqdm import tqdm

'''
kdtree kernel tests
---

'''

def plot_pcl(points, colors = None):
  fig = plt.figure()
  ax = fig.add_subplot()

  N = points.elem

  points = points.cpu().numpy(soil.index([N]))
  xs = points[:, 0]
  ys = points[:, 1]
  zs = points[:, 2]

#  levels = np.linspace(zs.min(), zs.max(), 32)
  # ax.tricontourf(xs, ys, zs, levels=levels, cmap='turbo')
  col = colors.cpu().numpy(soil.index([N]))
  ax.plot(xs, ys, 'o', markersize=2, color='grey')
  ax.tripcolor(xs, ys, col, shading='gouraud')

  plt.show()

#    print()
  '''

  if not colors is None:
    col = colors.cpu().numpy(soil.index([N]))
    ax.scatter(xs, ys, marker='o', c = col)
  else:
    ax.scatter(xs, ys, marker='o')

  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  plt.show()
  '''

'''

'''

def plot_pcl_3D(points, colors = None):

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  N = points.elem

  points = points.cpu().numpy(soil.index([N]))
  xs = points[:, 0]
  ys = points[:, 1]
  zs = points[:, 2]

#    print()

  if not colors is None:
    col = colors.cpu().numpy(soil.index([N]))
    print(np.max(col))
    col = np.log(1.0 + col)
    ax.scatter(xs, ys, zs, marker='o', c = col)
  else:
    ax.scatter(xs, ys, zs, marker='o')

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

    # Load Digital Elevation Model
    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu()

    # Sample Random Positions in 2D,
    # lerp the height-map to get the corresponding height-values
    # concatenate these into a point-cloud map!

    pos = soil.sampleN(index, 8192)
    kdtree = soil.kdtree(pos)
    height = soil.sample_lerp(buffer, index, pos)
    pcl = soil.concat(pos, height)

    print("Computing Accumulation...")
    acc = soil.sparseacc(kdtree, pcl, index, 1024)

    plot_pcl(pcl, acc)
#    plot_pcl_3D(pcl, acc)
    return

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)