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
    '''

    pcl = soil.pointcloud_sample(buffer, index, 8192)
#    pcl = pcl.cpu().numpy(soil.index([8192]))

    kdtree = soil.kdtree(pcl)

    query = soil.buffer.from_numpy(np.array([
      [0.0, 0.0, 0.0],
      [10.0, 10.0, 10.0]
    ], dtype=np.float32)).gpu()

    result = kdtree.knn(query, 5)
    result = result.cpu().numpy(soil.index([2, 5]))
    print(result)
  

#    print(pcl)
    return




    '''
    Optimization Procedure:
    1. Downsample Image
    2. Add Layers of RBF up to Resolution
    3. Jointly Optimize on Next Image Resolution
    4. Add Layer at that Resolution
    5. Repeat?

    steps = 4096
    layers = [
      RBFLayer(gencenters(1, tshape).to(device='cuda')),
      RBFLayer(gencenters(2, tshape).to(device='cuda')),
      RBFLayer(gencenters(4, tshape).to(device='cuda')),
      RBFLayer(gencenters(8, tshape).to(device='cuda')),
      RBFLayer(gencenters(16, tshape).to(device='cuda')),
      RBFLayer(gencenters(32, tshape).to(device='cuda')),
      RBFLayer(gencenters(64, tshape).to(device='cuda')),
    ]

    # Downsample Image

    print("Optimizing Layers Individually...")

    pos, val = downsample(buffer, (32, 32))
    layers[0].fit(pos, val, steps)
 
    buffer = buffer - layers[0].full(index)
    pos, val = downsample(buffer, (32, 32))
    layers[1].fit(pos, val, steps)

    buffer = buffer - layers[1].full(index)
    pos, val = downsample(buffer, (32, 32))
    layers[2].fit(pos, val, steps)

    buffer = buffer - layers[2].full(index)
    pos, val = downsample(buffer, (32, 32))
    layers[3].fit(pos, val, steps)

    buffer = buffer - layers[3].full(index)
    pos, val = downsample(buffer, (32, 32))
    layers[4].fit(pos, val, steps)

    # Jointly Optimize Layers 

    print("Optimizing Layers Jointly...")

    rbf = RBFInterpolator([
      layers[0],
      layers[1],
      layers[2],
      layers[3],
      layers[4]
    ])
    buffer = image.buffer.gpu().torch(index)
    pos, val = downsample(buffer, (64, 64))
    rbf.fit(pos, val, 1024)

    # Optimize Individually Again

    print("Optimizing Next Scale...")

    buffer = buffer - rbf.full(index)
    pos, val = downsample(buffer, (64, 64))
    layers[5].fit(pos, val, steps)

    buffer = buffer - rbf.full(index)
    pos, val = downsample(buffer, (128, 128))
    layers[6].fit(pos, val, 512)

    print("Optimizing Layers Jointly...")

    rbf = RBFInterpolator([
      layers[0],
      layers[1],
      layers[2],
      layers[3],
      layers[4],
      layers[5]
    ])
    buffer = image.buffer.gpu().torch(index)
    pos, val = downsample(buffer, (128, 128))
    rbf.fit(pos, val, 4096)

    '''


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