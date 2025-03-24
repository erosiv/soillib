#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np
import torch
from tqdm import tqdm


'''
PCA-Tree:
We wish to identify oriented gaussians that comprise our image. What this does
is it helps correctly identify the centroids of our data, and helps us build a
kind of binary tree. We will construct and then analyze this structure's shape.

We can potentially use this as a base structure for fitting RBF or forward-
generating terrain models.
'''

def fullimage(index):
  posx, posy = torch.meshgrid(
    torch.linspace(0, index[0]-1, index[0]),
    torch.linspace(0, index[1]-1, index[1]),
    indexing='ij')
  pos = torch.stack((posx, posy), dim=-1)
  return pos.to(device='cuda')

class PCANode:

  @staticmethod
  def mean(pos, val):
    return torch.sum(val.unsqueeze(-1)*pos, axis=0)/torch.sum(val)

  @staticmethod
  def cov(pos, val):
    return torch.cov(pos.T, aweights=val)

  def __init__(self, pos, val):

    '''
    We need the positions and the values...
    '''

    self.N = val.numel()

    self.min = torch.min(val)
    self.max = torch.max(val)

    self.pos = pos  # Positions is raw positions buffer
    self.val = val  # Note: Values act as weights

    self.val = (self.val - self.min)/(self.max - self.min)
#    self.val = torch.pow(self.val + 0.0001, 1.1)
#    self.val = 1.0 / (self.val + 0.0001)

#    print(self.pos.shape)
#    print(self.val.shape)

    # Mean and Covariance Matrix

    self.mean = PCANode.mean(self.pos, self.val)
    self.cov = PCANode.cov(self.pos-self.mean, self.val)

    # Eigendecomposition

    L, V = torch.linalg.eig(self.cov)
    self.EW = L.to(dtype=torch.float32)
    self.EV = V.to(dtype=torch.float32)

    ind = 1
    if(self.EW[1] > self.EW[0]):
      ind = 0
#      print("ZERO")
#    else:
#      print("ONE")

    # Split the Centroids

    self.nodeA = None
    self.nodeB = None

    splitA = (torch.sum((self.pos-self.mean)*self.EV[ind], axis=-1) < 0.0)
    splitB = ~splitA

    countA = torch.sum(splitA).cpu().item()
    countB = torch.sum(splitB).cpu().item()
    if(countA < 256): return
    if(countB < 256): return

    self.nodeA = PCANode(self.pos[splitA], self.val[splitA])
    self.nodeB = PCANode(self.pos[splitB], self.val[splitB])

  def centroids(self):
    mean = self.mean.cpu().numpy()
    if self.nodeA is None and self.nodeB is None:
      return np.array([
        mean,
      ])
    else:
      return np.array([
# note: this only shows centroids at the lowest levels
# for all levels, uncomment the mean here
#        mean,
        *self.nodeA.centroids(),
        *self.nodeB.centroids()
      ])

  def graph(self):
    if self.nodeA is None and self.nodeB is None:
      return np.array([])

    mean = self.mean.cpu().numpy()
    meanA = self.nodeA.mean.cpu().numpy()
    meanB = self.nodeB.mean.cpu().numpy()
    
    yA = [mean[0], meanA[0]]
    xA = [mean[1], meanA[1]]
    yB = [mean[0], meanB[0]]
    xB = [mean[1], meanB[1]]

    return np.array([
      xA, yA, xB, yB, 
      *self.nodeA.graph(),
      *self.nodeB.graph()
    ])

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu().torch(index)
    tshape = torch.Tensor([index[0], index[1]]).to(device='cuda')
    N = index[0]*index[1]

    # Construct the PCATree
    print("Sampling Data-Values...")
    pos = fullimage(index).view((N, 2))
    val = buffer.view((N))

    print("Constructing PCANode...")
    node = PCANode(pos, val)

    '''
    Visualization
    '''

    print("Visualizing...")
#    buffer = image.buffer.gpu().torch(index)
#    height_new = newimage.cpu().numpy()
    height = image.buffer.cpu().numpy(index)
    vmin = np.min(height)
    vmax = np.max(height)

    fig, axs = plt.subplots(1)
    axs.imshow(height, cmap='rainbow', vmin=vmin, vmax=vmax)
    
    centroids = node.centroids()

    print(centroids.shape)
    axs.scatter(centroids[:,1], centroids[:,0])
    
#    meanA = node.nodeA.mean.cpu().numpy()
#    meanB = node.nodeB.mean.cpu().numpy()

#    axs.plot([meanA[1], meanB[1]], [meanA[0], meanB[0]])

    '''
#    np.random.seed(0)
    n = 32
    a = np.random.uniform(0, 512, (n, 2))
    b = np.random.uniform(0, 512, (n, 2))

 #   fig, ax = plt.subplots(figsize=(3, 3))
    print(a)
    print(b)
    ab_pairs = np.c_[a, b]
    ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)

    # segments
    print(ab_args)
    axs.plot(*ab_args, c='k')

    # identify points: a in blue, b in red
#    axs.plot(*a.T, 'bo')
#    axs.plot(*b.T, 'ro')
    '''

    axs.plot(*node.graph(), c = 'k')






    plt.show()  

    
    '''
    pos, val = downsample(buffer, (128, 128))
    rbf.fit(pos, val, 4096)

    buffer = image.buffer.gpu().torch(index)
    rbf = RBFInterpolator(layers)
    buffer = resize(buffer, (128, 128))
    height = buffer.cpu().numpy()

    newimage = rbf.full(index)
    newimage = resize(newimage, (128, 128))
    '''

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)