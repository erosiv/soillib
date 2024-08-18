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

  normal = soil.normal()(model.height.array())
  normal_data = normal.numpy()
  height_data = model.height.array().numpy()
  relief = relief_shade(height_data, normal_data)

  discharge_data = sigmoid(model.discharge.array().numpy())

  # Compute Shading
  fig, ax = plt.subplots(2, 2, figsize=(16, 16))
  ax[0, 0].imshow(discharge_data)
  ax[0, 1].imshow(height_data)
  ax[1, 0].imshow(relief, cmap='gray')
  ax[1, 1].imshow(normal_data)
  plt.show()

'''
Erosion Code
'''

def make_model(shape, seed=0.0):

  '''
  returns a model wrapper type,
  which contains a set of layer
  references required for the
  hydraulic erosion model.
  '''

  height = soil.array(soil.float32, shape).fill(0.0)

  noise = soil.noise()
  for pos in shape.iter():
    index = shape.flat(pos)
    value = noise.get([pos[0]/shape[0], pos[1]/shape[1], seed])
    height[index] = 80.0 * value

  height    = soil.cached(soil.float32, height)
  discharge = soil.cached(soil.float32, soil.array(soil.float32,  shape).fill(0.0))
  momentum  = soil.cached(soil.vec2,    soil.array(soil.vec2,     shape).fill([0.0, 0.0]))

  resistance =  soil.constant(soil.float32, 0.0)
  maxdiff =     soil.constant(soil.float32, 0.8)
  settling =    soil.constant(soil.float32, 1.0)

  return soil.water_model(
    shape,
    height,
    momentum,
    discharge,
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

    # Tracking Values:
    discharge_track = soil.array(soil.float32, model.shape).fill(0.0)
    momentum_track = soil.array(soil.vec2, model.shape).fill([0.0, 0.0])

    with soil.timer() as timer:

      for n in range(n_particles):

        # Random Particle Position
        pos = 512*np.random.rand(2)
        drop = soil.water(pos)

        # Descend Particle
        while(True):

          if not drop.move(model):
            break

          # Update Tracking Values:

          #oob = model.shape.oob(drop.pos)
          #test = ''
          #print(model.shape)
          if not model.shape.oob(drop.pos):
            index = model.shape.flat(drop.pos)

            discharge_track.add_float(index, drop.volume)
            momentum_track.add_vec2(index, drop.volume, drop.speed)

          if not drop.interact(model):
            break

        # Accumulate Exit Fraction

        if model.shape.oob(drop.pos):
          no_basin_track += 1

      # Update Fields...
      # Execute the Tracking Update!!!

      # Update Trackable Quantities:
      model.discharge.array().track_float(discharge_track, lrate)
      model.momentum.array().track_vec2(momentum_track, lrate)

    exit_frac = (no_basin_track / n_particles)
    print(f"{step} ({exit_frac:.3f})")
    yield model.height, model.discharge

def main():

  np.random.seed(0)
  shape = soil.shape([512, 512])
  model = make_model(shape, seed = 1.0)
  for h, d in erode(model, steps = 750):
    pass

  render(model)

if __name__ == "__main__":
  main()