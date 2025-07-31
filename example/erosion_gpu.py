#!/usr/bin/env python

import soillib as soil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def noise(shape, scale):
  noise_param = soil.noise_t()
  noise_param.ext = np.array([shape[0], shape[1]]) * scale
  noise_param.seed = 3
  tensor = soil.noise(shape, noise_param)
  soil.multiply(tensor, 1.0)
  return tensor.gpu()

def full(value, shape, dtype=soil.float32, host=soil.cpu):
  tensor = soil.tensor(dtype, shape, host)
  soil.set(tensor, value)
  return tensor

def load_png(filename):
  im_frame = Image.open(filename)
  uplift = np.array(im_frame.getdata()).reshape(1024, 1024, 3) / 255.0
  uplift = uplift[:,:,0]
  tensor = soil.tensor.from_numpy(uplift.astype(np.float32)).gpu()
  return tensor

def main():

  '''
  Simulation Resolution
  '''

  simres = np.array([1024, 1024])       # Resolution [px]
  shape = soil.shape(*simres)           # Shape
  wscale = np.array([20.0, 20.0, 4.0])  # World Scale [km] (x, y, z)
  nscale = np.array([20.0, 20.0])       # Noise Feature Scale [km] (x, y)
  pscale = [wscale[0]/simres[0],        # Pixel Scale [km/px]
            wscale[1]/simres[1],
            wscale[2]]                  # Value Scale [km/unit]

  '''
  Simulation Model Setup
  '''

  # Overall Model
  model = soil.map_t(shape, pscale)
  model.height = full(0.0, shape, dtype=soil.float32, host=soil.gpu)
  #model.height = noise(shape, nscale / wscale[0:2])
  model.sediment = full(0.0, shape, dtype=soil.float32, host=soil.gpu)
  model.rainfall = full(1.0, shape, dtype=soil.float32, host=soil.gpu)
  model.uplift = load_png('C:/Users/nicho/Datasets/uplift_maps/uplift_blur.png')

#  model.uplift = full(0.0, shape, dtype=soil.float32, host=soil.gpu)

  # Tranported Data
  data = soil.data_t(shape)
  data.discharge = full(0.0, shape, dtype=soil.float32, host=soil.gpu)
  data.mass      = full(0.0, shape, dtype=soil.float32, host=soil.gpu)
  data.debris    = full(0.0, shape, dtype=soil.float32, host=soil.gpu)
  data.momentum        = full(0.0, soil.shape(*simres, 2), dtype=soil.float32, host=soil.gpu)
  data.debris_momentum = full(0.0, soil.shape(*simres, 2), dtype=soil.float32, host=soil.gpu)

  track = soil.data_t(shape)
  track.discharge = full(0.0, shape, dtype=soil.float32, host=soil.gpu)
  track.mass      = full(0.0, shape, dtype=soil.float32, host=soil.gpu)
  track.debris    = full(0.0, shape, dtype=soil.float32, host=soil.gpu)
  track.momentum        = full(0.0, soil.shape(*simres, 2), dtype=soil.float32, host=soil.gpu)
  track.debris_momentum = full(0.0, soil.shape(*simres, 2), dtype=soil.float32, host=soil.gpu)

  # Construct Parameters

  param = soil.param_t()
  param.timeStep = 10.0 # Geological Timestep [y]
  param.samples = 8192  # Number of Patricle Samples
  param.maxage = 256    # Maximum Particle Lifetime
  param.lrate = 1.0     # Filter Learning Rate

  param.gravity = 9.81    # Specific Gravity [m/s^2]
  param.uplift = 0.01     # Uplift Rate [m/y]
  
  param.rainfall = 1.0              # Rainfall Rate [m/y]
  param.evapRate = 0.0005           # Evapotranspiration Rate [1/s]
  param.viscosity = 0.000001        # Water Viscosity [m^2/s]
  param.bedShear = 12.5             # Turbulent Shear Stress [Pa = kg/m/s^2]
  param.suspensionRate = 0.0000008  # Fluvial Suspension Rate
  param.depositionRate = 0.00001    # Fluvial Deposition Rate
  param.fluvialExponent = 0.01      # Fluvial Power Exponent
  param.exitSlope = 0.025           # Slope Boundary Condition

  param.critSlope = 0.57                # Critical Slope
  param.debrisCreepRate = 0.0025        # Landslide Erosion Rate
  param.debrisSuspensionRate = 0.00025  # Debris Suspension Rate
  param.debrisDepositionRate = 0.0001   # Debris Deposition Rate
  param.debrisYieldStress = 2E6         # Yield Stress [Pa]
  param.debrisDensity = 2500.0          # Debris Density [kg/m^3]
  param.debrisViscosity = 0.004         # Debris Viscosity [m^2/s]
  param.debrisBedShear = 60/2500.0      # Debris Turbulent Shear Stress

  timer = soil.timer()
  for i in range(2048):
    with timer:
      soil.erode(model, data, track, param, 1)
    print(f"Execution Time: {timer.count}ms")

#  tiff_out = soil.tiff(height.cpu(), index)
#  tiff_out.write("/home/nickmcdonald/Datasets/erosion_gpu.tiff")

#  soil.util.show_discharge(data.discharge.cpu())
  soil.util.show_height(model.height.cpu())
  plt.show()

if __name__ == "__main__":
  main()