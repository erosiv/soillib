#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

'''
Visualization Code:
  Basic Relief Shade from Height, Normal
  w. Matplotlib Plotting (Float64)
'''

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

  normal = soil.normal()(model.height)
  normal_data = np.array(normal)
  height_data = np.array(model.height)

  # Compute Shading
  relief = relief_shade(height_data, normal_data)
  print(relief.shape, relief.dtype)
  plt.imshow(relief, cmap='gray')
  plt.show()

'''
Erosion Code
'''

def make_model(shape):

  '''
  returns a model wrapper type,
  which contains a set of layer
  references required for the
  hydraulic erosion model.
  '''

  height = soil.array("float", shape).fill(0.0)  
  discharge = soil.constant("float", 0.0)
  momentum =  soil.constant("vec2", [0.0, 0.0])
  resistance = soil.constant("float", 0.0)

  maxdiff = soil.constant("float", 0.8)
  settling = soil.constant("float", 1.0)

  return soil.water_model(
    shape,
    height,
    momentum,
    discharge,
    resistance,
    maxdiff,
    settling
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

    for n in range(n_particles):

      # Random Particle Position
      pos = 512*np.random.rand(2)
      drop = soil.water(pos)

      # Descend Particle
      while(True):

        if not drop.move(model):
          break

        # ... update the tracking maps ...
        # discharge_track[particle.pos] += particle.volume
        # momentum_track[particle.pos] += particle.volume * particle.speed

        if not drop.interact(model):
          break

      # Accumulate Exit Fraction
      if model.shape.oob(drop.pos):
        no_basin_track += 1

      # Update Fields...
      # Execute the Tracking Update!!!

      # Trackable / Updatable Layers!
      # Note: The Tracking Quantities should be 
      #discharge = soil.array("float", shape).fill(0.0)
      #discharge_track = soil.array("float", shape).fill(0.0)

      #momentum = soil.array("vec2", shape).fill([0.0, 0.0])
      #momentum_track = soil.array("vec2", shape).fill([0.0, 0.0])

    exit_frac = (no_basin_track / n_particles)
    print(f"{step} ({exit_frac:.3f})")

def main():

  shape = soil.shape([512, 512])  # Define Map Shape
  model = make_model(shape)       # Construct Model

  # Initial Condition

  noise = soil.noise()
  for pos in shape.iter():
    index = shape.flat(pos)
    value = noise.get([pos[0]/shape[0], pos[1]/shape[1], 0.1])
    model.height[index] = 80.0 * value

  # Run Erosion Code

  erode(model, steps = 256)
  render(model)

if __name__ == "__main__":
  main()