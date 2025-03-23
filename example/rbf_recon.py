#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np
import torch
from tqdm import tqdm

'''
Multi-Scale RBF Fitting:

In order to fit the DEM properly, we have to make sure that the fitting
process is correctly multi-scale. This means that we have to remove features
at the correct scales progressively while using all the image data at various
resolutions.

This means that we have to sample down the image to various resolutions, and
progressively subtract the approximations at various levels so that we don't
have to repeat the addition process all the time (for peformance).

A number of utility functions will be necessary to make this efficient.

Finally, we can attempt to optimize the RBF computation performance.
'''

def fullimage(index):
  # Sample Image Dataset
  posx, posy = torch.meshgrid(
    torch.linspace(0, index[0]-1, index[0]),
    torch.linspace(0, index[1]-1, index[1]),
    indexing='ij')
  pos = torch.stack((posx, posy), dim=-1)
  return pos.to(device='cuda')

def sparseimage(index, K = 1024):
  shape = torch.Tensor([index[0], index[1]])
  return (torch.rand(K, 2)*shape).to(device='cuda')

def sampleimage(image, index, pos):
  ipos = pos.to(dtype = torch.int)
  test = image[ipos[..., 0], ipos[..., 1]]
  return test

def resize(image, size):
  view = image.unsqueeze(-1).permute(2, 0, 1).unsqueeze(0)
  view = torch.nn.functional.interpolate(view, size=size) # mode='bilinear'
  return view[0].permute(1, 2, 0).squeeze(-1)

def fmap(dist, shape):
  return 1.0 / (1.0 + (shape*shape*dist*dist))

class RBFLayer:

  def __init__(self, centers):

    self.centers = centers
    self.K = self.centers.shape[0]
    self.weights = (torch.full((self.K,), 0.0)).to(device='cuda')
    self.shapes = (torch.full((self.K,), 0.02)).to(device='cuda')

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

  def fit(self, pos, val, steps = 2048):

    params = self.params()
    optimizer = torch.optim.Adam([
      {'params': [params[0]], 'lr': 1e-2},
      {'params': [params[1]], 'lr': 1e-2},
      {'params': [params[2]], 'lr': 1e-3},
    ])

    loss = None
    with tqdm(range(steps)) as t:
      for step in t:
        optimizer.zero_grad()
        diff = self.sample(pos.detach()) - val.detach()
        loss = torch.mean(diff*diff)
        loss.backward()
        optimizer.step()
    print(f"Loss ({loss.item():.6f})")

    for param in params:
      param.requires_grad = False

class RBFInterpolator:

  def __init__(self, layers):
    self.layers = layers

  def sample(self, pos):
    approx = torch.full(pos.shape[:-1], 0.0, device='cuda')
    for layer in self.layers:
      approx += layer.sample(pos)
    return approx

  def full(self, index):
    image = torch.full((index[0], index[1]), 0.0, device='cuda')
    for layer in self.layers:
      image += layer.full(index)
    return image

  def fit(self, pos, val, steps = 2048):

    pweights = []
    pcenters = []
    pshapes = []

    for layer in self.layers:
      params = layer.params()
      pweights = [*pweights, params[0]]
      pcenters = [*pcenters, params[1]]
      pshapes = [*pshapes, params[2]]
    
    optimizer = torch.optim.Adam([
      {'params': pweights, 'lr': 1e-2},
      {'params': pcenters, 'lr': 1e-2},
      {'params': pshapes, 'lr': 1e-3},
    ])

    loss = None
    with tqdm(range(steps)) as t:
      for step in t:
        optimizer.zero_grad()
        approx = self.sample(pos)
        loss = torch.mean((val-approx)*(val-approx))
        loss.backward()
        optimizer.step()
    print(f"Loss ({loss.item():.6f})")

    for param in pweights:
      param.requires_grad = False
    for param in pcenters:
      param.requires_grad = False
    for param in pshapes:
      param.requires_grad = False

def downsample(buffer, shape):
  shape = torch.Size(shape)
  scale = torch.Tensor([buffer.shape[0]/shape[0], buffer.shape[1]/shape[1]]).to(device='cuda')
  pos = fullimage(shape)*scale
  val = resize(buffer, shape)
  return pos, val

def gencenters(N, shape):
  centers = (1 + fullimage((N, N)))/(N+1)
  centers = centers * shape.unsqueeze(0).unsqueeze(0)
  centers = centers.view((N*N, 2))
  return centers

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu().torch(index)
    tshape = torch.Tensor([index[0], index[1]]).to(device='cuda')

    '''
    Optimization Procedure:
    1. Downsample Image
    2. Add Layers of RBF up to Resolution
    3. Jointly Optimize on Next Image Resolution
    4. Add Layer at that Resolution
    5. Repeat?
    '''

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