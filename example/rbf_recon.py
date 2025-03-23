#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np
import torch
from tqdm import tqdm

def fullimage(index):
  # Sample Image Dataset
  posx, posy = torch.meshgrid(
    torch.linspace(0, index[0]-1, index[0]),
    torch.linspace(0, index[1]-1, index[1]),
    indexing='ij')
  pos = torch.stack((posx, posy), dim=-1)
  print(pos.shape)
  return pos

def sparseimage(index, K = 1024):
  shape = torch.Tensor([index[0], index[1]])
  return torch.rand(K, 2)*shape

def sampleimage(image, index, pos):
  test = image[pos[:, 0], pos[:, 1]]
  return test

class RBFInterpolator:

  def __init__(self, shape, K):

    self.K = K
    self.shape = shape

    self.centers = (torch.rand(K, 2)*self.shape).to(device='cuda')
    self.weights = (torch.full((K,), 0.0)).to(device='cuda')
    self.shapes = (torch.full((K,), 0.02)).to(device='cuda')

  def sample(self, pos):

    def fmap(dist, shape):
      return 1.0 / (1.0 + shape * dist)

    dist = torch.cdist(pos, self.centers, p = 2)
    vals = fmap(dist, self.shapes)
    return torch.matmul(vals, self.weights)

  def fit(self, pos, val, lr = 0.01, steps = 4096):

    params = [self.weights, self.centers]
    for param in params:
      param.requires_grad = True

    optimizer = torch.optim.Adam(params, lr = lr) # Optimizer w. Learning Rate

    for step in tqdm(range(steps)):
      
      optimizer.zero_grad()
      approx = self.sample(pos)
      loss = torch.mean((val - approx) * (val - approx))
      loss.backward()
      optimizer.step()

      print("LOSS", loss.cpu().item())

  def full(self, index):
    with torch.no_grad():
      def fmap(dist, shape):
        return 1.0 / (1.0 + shape * dist)

      pos = fullimage(index).to(device='cuda')
      image = torch.full((index[0], index[1]), 0.0, device='cuda')

      for k, center in tqdm(enumerate(self.centers)):
        dist = torch.cdist(pos, center.unsqueeze(0), p = 2)
        vals = fmap(dist, self.shapes[k])
        image += vals.squeeze(-1)*self.weights[k]
  #      print(vals.shape)
  #      image += torch.matmul(vals, self.weights)

      return image

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index

    # Construct RBF Interpolator (Sparse Support Fit)
    buffer = image.buffer.gpu().torch(index)
    pos = sparseimage(index, 4096).to(dtype=torch.int, device='cuda')
    val = sampleimage(buffer, index, pos)
    pos = pos.to(dtype=torch.float32)
    shape = torch.Tensor([index[0], index[1]])
    rbf = RBFInterpolator(shape, 1024)
    rbf.fit(pos, val)

    # Reconstruct Full Image from RBFInterpolator

#    print(buffer)
#    print(newimage)
#    return

    '''
    Visualization Code
    '''

    newimage = rbf.full(index)
    height_new = soil.buffer.from_numpy(newimage.cpu().numpy())
    normal_new = soil.normal(height_new, index, image.meta.scale).numpy(index)
    height_new = height_new.numpy(index)

    height = image.buffer.cpu().numpy(index)
    normal = soil.normal(image.buffer, index, image.meta.scale).numpy(index)

    # Compute Shading
    relief_new = relief_shade(height_new, normal_new)
    relief = relief_shade(height, normal)

    fig, axs = plt.subplots(2)
    #fig.suptitle('Vertically stacked subplots')
    axs[0].imshow(relief_new, cmap='gray')
    axs[1].imshow(relief, cmap='gray')

    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)