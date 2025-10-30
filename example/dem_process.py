#!/usr/bin/env python

import soillib as soil
import silt

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

'''
Example Code Showcasing Various
DEM Processing Kernels
'''

def load(data):

  image = soil.geotiff(data)
  print(f"Loaded File: {data}, {image.buffer.type}, ({image.index[0]}, {image.index[1]})")
  return (image.buffer, image.index)

def discharge_fastflow(tensor):

  shape = tensor.shape
  res = (shape[0], shape[1])

  rain = np.full(res, 1.0)
  rain = silt.tensor.from_numpy(rain.astype(np.float32)).gpu()

  t = soil.timer(soil.us)
  with t:
    dirn = soil.direction(tensor, soil.d8)
    # flow = soil.steepest(tensor, soil.d8)
    flow = soil.random_weighted(tensor, soil.d8, 0, 0, 10.0)
    discharge = soil.accumulate(flow, rain, soil.d8)
  print(f"Execution Time: {t.count} us")

  return discharge.cpu().numpy()

def discharge_stochastic(tensor):

  shape = tensor.shape
  res = (shape[0], shape[1])

  rain = np.full(res, 1.0)
  evap = np.full(res, 0.001)

  rain = silt.tensor.from_numpy(rain.astype(np.float32)).gpu()
  evap = silt.tensor.from_numpy(evap.astype(np.float32)).gpu()

  k = 8192*48
  rng = silt.tensor(silt.rng, silt.shape(k), silt.gpu)
  silt.seed(rng, 0, 0)

  scale = [0.01, 0.01]

#  grad = grad.cpu().numpy()
#  plt.imshow(grad[..., 1])
#  plt.show()

  # Diffuse the Tensor...
#  for i in range(50):
#    diff = soil.laplacian(tensor, [1.0, 1.0])
#    silt.multiply(diff, 0.5)
#    silt.add(tensor, diff)
#  plt.imshow(tensor.cpu().numpy())
#  plt.show()
#  tensor.gpu()

#  t = soil.timer(soil.us)
#  with t:

  # Diffusion of the velocity field works...
  #   We need to fix the numerical stability
  #   and make sure that the scale is implemented correctly.
  #   
  grad = soil.gradient(tensor, scale)
#  for i in range(5000):
#    diff = soil.laplacian(grad, [1.0, 1.0])
#    silt.multiply(diff, 0.2)
#    silt.add(grad, diff)

  silt.multiply(grad, -1)
  discharge = soil.solve_uniform(grad, rain, evap, rng, scale, k)

#  print(f"Laplacian Shape: {laplacian.shape}")
#  laplacian = laplacian.cpu().numpy()
#  grad = grad.cpu().numpy()
#  plt.imshow(grad[..., 1])
#  plt.show()

#  print(f"Execution Time: {t.count} us")

  return discharge.cpu().numpy()

def main(data):

  # Load GeoTIFF Data
  tiff = soil.geotiff(data)
  tensor = tiff.tensor.gpu()

  discharge = discharge_stochastic(tensor)
#  discharge = discharge_fastflow(tensor)

  plt.imshow(discharge,
    cmap='CMRmap',
    norm=colors.LogNorm(1, discharge.max()),
    interpolation='none'
  )
  plt.show()

if __name__ == "__main__":

  data = "C:\\Users\\nicho\\Datasets\\test.tiff"
  #data = "_dem_conditioned.tiff"

  main(data)