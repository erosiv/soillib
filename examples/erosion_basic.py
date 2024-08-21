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

  normal = soil.normal(index, model.height).full()
  normal_data = normal.numpy().reshape((index[0], index[1], 3))
  height_data = model.height.buffer().numpy().reshape((index[0], index[1]))
  relief = relief_shade(height_data, normal_data)

  discharge_data = sigmoid(model.discharge.buffer().numpy().reshape((index[0], index[1])))

  momentum_data = sigmoid(model.momentum.buffer().numpy().reshape((index[0], index[1], 2)))
  momentum_data = np.append(momentum_data, np.zeros((512, 512, 1)), axis=-1)

  # Compute Shading
  fig, ax = plt.subplots(2, 2, figsize=(16, 16))
  ax[0, 0].imshow(discharge_data)
  #ax[0, 1].imshow(height_data)
  ax[0, 1].imshow(momentum_data)
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

  height = soil.buffer(soil.float32, index.elem()).fill(0.0)
  
  noise = soil.noise(index, seed)
  height = noise.full()

  for pos in index.iter():
    i = index.flatten(pos)
    height[i] = 80.0 * height[i]

  discharge = soil.buffer(soil.float32, index.elem()).fill(0.0)
  momentum  = soil.buffer(soil.vec2,    index.elem()).fill([0.0, 0.0])

  discharge_track = soil.buffer(soil.float32, index.elem()).fill(0.0)
  momentum_track  = soil.buffer(soil.vec2,    index.elem()).fill([0.0, 0.0])

  resistance  = soil.constant(soil.float32, 0.0)
  maxdiff     = soil.constant(soil.float32, 0.8)
  settling    = soil.constant(soil.float32, 1.0)

  return soil.water_model(
    index,
    soil.layer(height),
    soil.layer(momentum),
    soil.layer(momentum_track),
    soil.layer(discharge),
    soil.layer(discharge_track),
    soil.layer(resistance),
    soil.layer(maxdiff),
    soil.layer(settling)
  )

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

  for step in range(steps):

    # Fraction of "Exited" Particles
    no_basin_track = 0.0
    model.discharge_track.buffer().zero()
    model.momentum_track.buffer().zero()

    # Tracking Values:

    with soil.timer() as timer:

      for n in range(n_particles):

        # Random Particle Position
        pos = 512*np.random.rand(2)
        drop = soil.water(pos)

        # Descend Particle
        while(True):

          if not drop.move(model):
            break

          drop.track(model)

          if not drop.interact(model):
            break

        if model.index.oob(drop.pos):
          no_basin_track += 1

      # Update Trackable Quantities:
      model.discharge.track_float(model.discharge_track, lrate)
      model.momentum.track_vec2(model.momentum_track, lrate)

    exit_frac = (no_basin_track / n_particles)
    print(f"{step} ({exit_frac:.3f})")
    yield model.height, model.discharge

def main():

  np.random.seed(0)
  index = soil.index([512, 512])
  model = make_model(index, seed = 16.0)
  for h, d in erode(model, steps = 1024):
    pass

  render(model)

if __name__ == "__main__":
  main()