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

'''
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

def diffuse(tensor, scale, dt):
  diff = soil.laplacian(tensor, scale)
  silt.multiply(diff, dt)
  silt.add(tensor, diff)
'''

# Determine the Erosion Source Term...
# Accumulate the Erosion Source Term...
# Decay by the Erosion Decay Term...
# Then we should get our material distribution.

def discharge_stochastic(tensor):

  shape = tensor.shape
  res = (shape[0], shape[1])
  scale = [0.01, 0.01]
  A = scale[0] * scale[1]

  rain = np.full(res, A)
  evap = np.full(res, 0.0)
  visc = np.full(res, 0.2)

  rain = silt.tensor.from_numpy(rain.astype(np.float32)).gpu()
  evap = silt.tensor.from_numpy(evap.astype(np.float32)).gpu()
  visc = silt.tensor.from_numpy(visc.astype(np.float32)).gpu()

  k = 8192*32
  rng = silt.tensor(silt.rng, silt.shape(k), silt.gpu)

  grad = soil.gradient(tensor, scale)
  silt.multiply(grad, -1)
  velocity = grad
  
  for i in range(64):

    # Accumulate Discharge
    silt.seed(rng, 0, i*k)
    discharge = soil.solve_uniform(velocity, rain, evap, rng, scale, k)

    # Accumulate Eroded Mass Downstream...
    # 
    silt.seed(rng, 0, i*k)
    suspend = soil.suspend(velocity, scale)# basically the shear-stress expression
    mass = soil.solve_uniform(velocity, suspend, evap, rng, scale, k)

    '''
    Erode the Surface ...
    Update the Gradient ...
    '''

#    suspend = mass_source.cpu().numpy()
    deposit = silt.tensor.from_numpy(mass.cpu().numpy() / discharge.cpu().numpy()).gpu()
    silt.multiply(suspend, -100)
    silt.multiply(deposit, 0.0)
    silt.add(deposit, suspend)

    silt.add(tensor, deposit)
#    silt.add(tensor, suspend)

    grad = soil.gradient(tensor, scale)
    silt.multiply(grad, -1)

    # Accumulate Momentum Downstream
    silt.seed(rng, 0, i*k)
    momentum_source = silt.clone(grad)
    silt.multiply(momentum_source, 0.0)
    momentum_source = silt.tensor.from_numpy(momentum_source.cpu().numpy() + rain.cpu().numpy()[..., np.newaxis]).gpu()
    silt.multiply(momentum_source, A)
    silt.add(momentum_source, grad)
    momentum = soil.solve_uniform(velocity, momentum_source, visc, rng, scale, k)
    velocity = silt.tensor.from_numpy(momentum.cpu().numpy() / discharge.cpu().numpy()[..., np.newaxis]).gpu()
    silt.multiply(velocity, A)
    rain.gpu()

    # Do the same thing for the other erosion type ...

  return discharge.cpu().numpy(), suspend.cpu().numpy()

def main(data):

  # Load GeoTIFF Data
  tiff = soil.geotiff(data)
  tensor = tiff.tensor.gpu()

  discharge, velocity = discharge_stochastic(tensor)
#  velocity = np.sum(np.abs(velocity), axis=2)

  print(f"Discharge Max: {np.max(discharge)}")
  print(f"Velocity MinMax: {np.min(velocity)}, {np.max(velocity)}")

  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  fig.suptitle("Grid-Free Monte-Carlo Estimator")

  ax[0].imshow(tensor.cpu().numpy())
#  ax[0].imshow(discharge,
#    cmap='CMRmap',
#    norm=colors.LogNorm(1, discharge.max()),
#    interpolation='none'
#  )
  ax[1].imshow(velocity,
    cmap='CMRmap',
    norm=colors.LogNorm(1, velocity.max()),
    interpolation='none'
  )
  plt.show()




#  t = soil.geotiff()
#  t.peek("my_output.tiff")

  tiff_out = soil.geotiff(tensor)
#  tiff_out.meta = t.meta
#  tiff_out.unsetnan()
  tiff_out.write("my_output.tiff")



if __name__ == "__main__":

  data = "C:\\Users\\nicho\\Datasets\\test.tiff"
  #data = "_dem_conditioned.tiff"

  main(data)