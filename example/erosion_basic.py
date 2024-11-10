#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

'''
Visualization Code:
  Basic Relief Shade from Height, Normal
  w. Matplotlib Plotting (Float64)
'''

lrate = 0.1

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def relief_shade(h, n):

  # Regularize Height
  h_min = np.nanmin(h)
  h_max = np.nanmax(h)
  h = (h - h_min)/(h_max - h_min)

  # Light Direction, Diffuse Lighting
  light = np.array([-1, 1, 2])
  light = light / np.linalg.norm(light)

  diffuse = np.sum(light * n, axis=-1)
  diffuse = 0.05 + 0.9*diffuse

  # Flat-Toning
  flattone = np.full(h.shape, 0.9)
  weight = 1.0 - n[:,:,2]
  weight = weight * (1.0 - h * h)

  # Full Diffuse Shading Value
  diffuse = (1.0 - weight) * diffuse + weight * flattone
  return diffuse

def render(model):

  index = model.index

  normal = soil.normal(index, model[soil.height]).full()
  normal_data = normal.numpy(index)

  height_data = soil.bake(model[soil.height], index).numpy(index)
  relief = relief_shade(height_data, normal_data)

  discharge_data = sigmoid(100.0*soil.bake(model[soil.discharge], index).numpy(index))
  #discharge_data = model[soil.discharge].numpy(index)

#  momentum_data = sigmoid(model.momentum.numpy(index))
#  momentum_data = np.append(momentum_data, np.zeros((512, 512, 1)), axis=-1)

  # Compute Shading
  fig, ax = plt.subplots(2, 2, figsize=(16, 16))
  ax[0, 0].imshow(discharge_data)
  ax[0, 1].imshow(height_data)
  #ax[0, 1].imshow(momentum_data)
  ax[1, 0].imshow(relief, cmap='gray')
  ax[1, 1].imshow(normal_data)
  plt.show()

'''
Erosion Code
'''

def make_model(index, seed=0.0):

  '''
  returns a model wrapper type,
  which contains a set of layer
  references required for the
  hydraulic erosion model.
  '''

  model = soil.model()
  model.index = index

  noise = soil.noise(index, seed)
  noise = soil.bake(noise, index)
  soil.multiply(noise, 80.0)
  model[soil.height] = soil.cached(noise)

  model[soil.discharge]       = soil.cached(soil.bake(soil.constant(soil.float32, 0.0), index))
  model[soil.momentum]        = soil.cached(soil.bake(soil.constant(soil.vec2, [0.0, 0.0]), index))
  model[soil.discharge_track] = soil.cached(soil.bake(soil.constant(soil.float32, 0), index))
  model[soil.momentum_track]  = soil.cached(soil.bake(soil.constant(soil.vec2, [0.0, 0.0]), index))

  model[soil.resistance] = soil.constant(soil.float32, 0.0)
  model[soil.maxdiff]    = soil.constant(soil.float32, 0.8)
  model[soil.settling]   = soil.constant(soil.float32, 1.0)

  return model

def erode(model, steps=512):

  '''
  Iterate over a maximum number of steps,
  spawn a set of particles, descend them,
  accumulate their properties and print
  the fraction that successfully exits the map.
  That fraction determines whether the
  "basins" have been solved yet or not.
  '''

  n_particles = 512
  no_basin_track = 0.0

  for step in range(steps):

    if no_basin_track >= 0.995:
      break;

    # Fraction of "Exited" Particles
    no_basin = 0.0
    model[soil.discharge_track][:] = 0.0
    model[soil.momentum_track][:] = [0.0, 0.0]

    # Tracking Values:

    with soil.timer() as timer:

      for n in range(n_particles):

        # Random Particle Position
        # Note: Random over total elements,
        # then unflatten the position.
        index = model.index.elem() * np.random.rand(1)
        pos = model.index.unflatten(int(index[0]))
        drop = soil.water(pos)

        # Descend Particle
        while(True):

          if not drop.move(model):
            break

          drop.track(model)

          if not drop.interact(model):
            break

        if model.index.oob(drop.pos):
          no_basin += 1

      # Update Trackable Quantities:
      model[soil.discharge].track(model[soil.discharge_track], lrate)
      model[soil.momentum].track(model[soil.momentum_track], lrate)

      no_basin = (no_basin / n_particles)
      no_basin_track = (1.0-lrate)*no_basin_track + lrate*no_basin
    
    # Print Information

    print(f"{step} ({no_basin_track:.3f})")
    yield model[soil.height], model[soil.discharge]

def main():

  np.random.seed(0)
  index = soil.index([512, 512])
  model = make_model(index, seed = 25.0)
  for h, d in erode(model, steps = 512):
    pass

  render(model)

if __name__ == "__main__":
  main()