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
  return (torch.rand(K, 2)*shape).to(device='cuda')

def sampleimage(image, index, pos):
  ipos = pos.to(dtype = torch.int)
  test = image[ipos[:, 0], ipos[:, 1]]
  return test

def fmap(dist, shape):
  return torch.exp(-shape*shape*dist*dist)

class RBFLayer:

  def __init__(self, index, K):

    self.K = K
    self.shape = torch.Tensor([index[0], index[1]])
    self.centers = (torch.rand(K, 2)*self.shape).to(device='cuda')
    self.weights = (torch.full((K,), 0.0)).to(device='cuda')
    self.shapes = (torch.full((K,), 0.02)).to(device='cuda')

  def sample(self, pos):

    dist = torch.cdist(pos, self.centers, p=2)
    vals = fmap(dist, self.shapes)
    return torch.matmul(vals, self.weights)

  def params(self):
    params = [self.weights, self.centers, self.shapes]
    for param in params:
      param.requires_grad = True
    return params

  def full(self, index):
    with torch.no_grad():
      pos = fullimage(index).to(device='cuda')
      image = torch.full((index[0], index[1]), 0.0, device='cuda')
      for k, center in tqdm(enumerate(self.centers)):
        dist = torch.cdist(pos, center.unsqueeze(0), p = 2)
        vals = fmap(dist, self.shapes[k])
        image += vals.squeeze(-1)*self.weights[k]
      return image

class RBFInterpolator:

  def __init__(self, index, K):
    self.layers = [ RBFLayer(index, k) for k in K ]

  def sample(self, pos):
    approx = torch.full((pos.shape[0],), 0.0, device='cuda')
    for layer in self.layers:
      approx += layer.sample(pos)
    return approx

  def full(self, index):
    image = torch.full((index[0], index[1]), 0.0, device='cuda')
    for layer in self.layers:
      image += layer.full(index)
    return image

  def fit(self, pos, val, layer, lr = 0.01, steps = 2048):

    params = self.layers[layer].params()
    optimizer = torch.optim.Adam([
      {'params': [params[0]], 'lr': 1e-2},
      {'params': [params[1]], 'lr': 1e-2},
      {'params': [params[2]], 'lr': 1e-5},
    ])

    with tqdm(range(steps)) as t:
      for step in t:
        optimizer.zero_grad(set_to_none=True)
        approx = self.sample(pos)
        loss = torch.mean((val - approx) * (val - approx))
        loss.backward()
        optimizer.step()
        t.set_description(f"Loss ({loss.item()})")

    for param in params:
      param.requires_grad = False

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu().torch(index)

    K = [4]#, 64, 128]
    rbf = RBFInterpolator(index, K)

    pos = sparseimage(index, 1024)
    val = sampleimage(buffer, index, pos)

    for i, k in enumerate(K):
      print(f"Converging Iteration {i}")
      rbf.fit(pos, val, i)









    '''
    Visualization Code
    '''

    newimage = rbf.full(index)
    height_new = newimage.cpu().numpy()
#    height_new = soil.buffer.from_numpy(newimage.cpu().numpy())
    #normal_new = soil.normal(height_new, index, image.meta.scale).numpy(index)
    #height_new = height_new.numpy(index)

    height = image.buffer.cpu().numpy(index)
    #normal = soil.normal(image.buffer, index, image.meta.scale).numpy(index)

    # Compute Shading
#    relief_new = relief_shade(height_new, normal_new)
#    relief = relief_shade(height, normal)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(height_new, cmap='gray')
    axs[1].imshow(height, cmap='gray')

    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/large_flat_texas.tiff"
  data = "/home/nickmcdonald/Datasets/erosion_gpu.tiff"

  main(data)