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
      {'params': [params[2]], 'lr': 1e-4},
    ])

    with tqdm(range(steps)) as t:
      for step in t:
        optimizer.zero_grad()
        approx = self.sample(pos)
        loss = torch.mean((val-approx)*(val-approx))
        loss.backward()
        optimizer.step()
        t.set_description(f"Loss ({loss.item():.3f})")

    for param in params:
      param.requires_grad = False

'''
class RBFInterpolator:

  def __init__(self, index, K):
    self.layers = [ RBFLayer(index, k) for k in K ]

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

  def fit(self, pos, val, layer, steps = 2048):

    params = self.layers[layer].params()
    optimizer = torch.optim.Adam([
      {'params': [params[0]], 'lr': 1e-2},
      {'params': [params[1]], 'lr': 1e-2},
      {'params': [params[2]], 'lr': 1e-3},
    ])

    with tqdm(range(steps)) as t:
      for step in t:
        optimizer.zero_grad()
        approx = self.sample(pos)
        loss = torch.mean((val-approx)*(val-approx))
        loss.backward()
        optimizer.step()
        t.set_description(f"Loss ({loss.item()})")

    for param in params:
      param.requires_grad = False
'''

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

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")
    index = image.index
    buffer = image.buffer.gpu().torch(index)

    steps = 4095
    layers = []

    # Downsample Image
    shape = (16, 16)
    buffer16 = resize(buffer, shape)
    pos = fullimage(shape)
    val = sampleimage(buffer16, shape, pos)

    # Lowest-Level Interpolatin Tensor
    centers = (1 + fullimage((1, 1)))/2
    centers = centers * 512
    print(centers.shape)

    rbf = RBFLayer(centers.to(device='cuda'))
    layers.append(rbf)

    rbf.fit(32*pos, val, steps)

    recon = rbf.full(index)
    buffer = buffer - recon

    # Downsample Image
    shape = (16, 16)
    buffer16 = resize(buffer, shape)
    pos = fullimage(shape)
    val = sampleimage(buffer16, shape, pos)

    centers = (1 + fullimage((2, 2)))/3
    centers = centers * 512
    centers = centers.view((4, 2))
    
    rbf = RBFLayer(centers.to(device='cuda'))
    layers.append(rbf)

    rbf.fit(32*pos, val, steps)

    recon = rbf.full(index)
    buffer = buffer - recon

    # Downsample Image
    shape = (16, 16)
    buffer16 = resize(buffer, shape)
    pos = fullimage(shape)
    val = sampleimage(buffer16, shape, pos)

    centers = (1 + fullimage((4, 4)))/5
    centers = centers * 512
    centers = centers.view((16, 2))
    print(centers.shape)

    rbf = RBFLayer(centers.to(device='cuda'))
    layers.append(rbf)

    rbf.fit(32*pos, val, steps)

    recon = rbf.full(index)
    buffer = buffer - recon

    # Downsample Image
    shape = (16, 16)
    buffer16 = resize(buffer, shape)
    pos = fullimage(shape)
    val = sampleimage(buffer16, shape, pos)

    centers = (1 + fullimage((8, 8)))/9
    centers = centers * 512
    centers = centers.view((64, 2))
    print(centers.shape)

    rbf = RBFLayer(centers.to(device='cuda'))
    layers.append(rbf)

    rbf.fit(32*pos, val, steps)

    recon = rbf.full(index)
    buffer = buffer - recon

    # Downsample Image
    shape = (16, 16)
    buffer16 = resize(buffer, shape)
    pos = fullimage(shape)
    val = sampleimage(buffer16, shape, pos)

    centers = (1 + fullimage((16, 16)))/17
    centers = centers * 512
    centers = centers.view((16*16, 2))
    print(centers.shape)

    rbf = RBFLayer(centers.to(device='cuda'))
    layers.append(rbf)

    rbf.fit(32*pos, val, steps)

    recon = rbf.full(index)
    buffer = buffer - recon

    '''
    Visualization Code
    '''





    rbf = RBFInterpolator(layers)
    newimage = rbf.full(index)
    height_new = newimage.cpu().numpy()
#    height_new = soil.buffer.from_numpy(newimage.cpu().numpy())
    #normal_new = soil.normal(height_new, index, image.meta.scale).numpy(index)
    #height_new = height_new.numpy(index)

#    resize_transform = transforms.Resize((16, 16))
    buffer = image.buffer.gpu().torch(index)
#    buffer = resize(buffer, (16, 16))
    height = buffer.cpu().numpy()

    #normal = soil.normal(image.buffer, index, image.meta.scale).numpy(index)

    # Compute Shading
#    relief_new = relief_shade(height_new, normal_new)
#    relief = relief_shade(height, normal)

    vmin = np.min(height)
    vmax = np.max(height)
    print("MEAN", np.mean(height))

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