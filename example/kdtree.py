#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np
from tqdm import tqdm

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu()

    '''
    Sample Pointcloud?

    Note: The pointcloud here is really a
      buffer which consists of a set of positions,
      and a buffer which consists of a set of height-values
      which are interpolated at those positions.

      So really if possible, we want to split that kernel...
      so that we just generate the positions and then write
      another kernel which lets that be sampled directly...

      Then we generate the height-field buffer by interpolation...
      and then we put this in the kd-tree in 2D, and we can
      sample the height-values like that.
    '''

    pos = soil.sampleN(index, 8192)
    height = soil.sample_lerp(buffer, index, pos)
    pcl = soil.concat(pos, height)

#    print(pos.cpu().numpy(soil.index([8192])))
#    print(height.cpu().numpy(soil.index([8192])))
#    print(pcl.cpu().numpy(soil.index([8192])))

    # do we want to concatenate? I suppose we can...
    # but I don't think that we really need it necessarily...
    kdtree = soil.kdtree(pcl)

    # can we query on a mesh-grid?
    # what's the performance of that?

    x = np.linspace(0, index[0]-1, index[0])
    y = np.linspace(0, index[1]-1, index[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.full(Y.shape, 0.5)
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()]).transpose().copy()
    positions = positions.astype(np.float32)

    print(positions)
    print(positions.shape)

    positions = np.array([
      [0.00e+00, 0.00e+00, 5.00e-01],
      [0.00e+00, 5.11e+02, 5.00e-01],
      [5.11e+02, 0.00e+00, 5.00e-01],
      [5.11e+02, 5.11e+02, 5.00e-01],
    ]).astype(np.float32)
  
    N = positions.shape[0]
    query = soil.buffer.from_numpy(positions).gpu()
    result = kdtree.knn(query, 5)

    nearest = soil.select_index(pos.gpu(), result.gpu())
    print(nearest.cpu().numpy(soil.index([N, 5])))

    result = result.cpu().numpy(soil.index([N, 5]))
  
    print(result)

    return

    '''
    Visualization Code
    '''

    buffer = image.buffer.gpu().torch(index)
    rbf = RBFInterpolator(layers)
    buffer = resize(buffer, (128, 128))
    height = buffer.cpu().numpy()

    newimage = rbf.full(index)
    newimage = resize(newimage, (128, 128))
    height_new = newimage.cpu().numpy()

    vmin = np.min(height)
    vmax = np.max(height)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(height_new, cmap='gray', vmin=vmin, vmax=vmax)
    axs[1].imshow(height, cmap='gray', vmin=vmin, vmax=vmax)

    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)