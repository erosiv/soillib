#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

'''
Basically, we construct an array with noise.
Then we create a bunch of "layers", which provide
the information that the model needs to work.

Then we pack that into a structure and let the particles
operate on it directly...

That should work right?
'''

def make_model(shape):

  height = soil.array("float", shape).fill(0.0)
  #discharge_volume = soil.array("float",).fill(0.0)
  
  discharge = soil.constant("float", 0.0)
  momentum =  soil.constant("fvec2", [0.0, 0.5])
  
  #print(momentum(0))
  
  #discharge = soil.layer(lambda d: erf(d))

  return None

  '''
  return soil.particle.water({
    "normal": soil.normal(data),
    "discharge": discharge,
    "discharge_track": discharge_track,
    "momentum": momentum,
    "momentum_track": momentum_track
  })
  '''

'''
def erode(n_cycles = 512):

  # Reset Tracking Buffers
  # Note: I don't like this, this should be stream-lined.
  discharge_track.fill(0.0)
  momentum_track.fill([0.0, 0.0])

  no_basin_track = 0.0

  for i in range(n_cycles):

    # spawn at random position in map...
    particle = soil.particle.water(...)

    while(True):

      if not particle.move(model):
        break

      # ... update the tracking maps ...
      discharge_track[particle.pos] += particle.volume
      momentum_track[particle.pos] += particle.volume * particle.speed

      if not particle.interact(model):
        break

    if oob(particle.pos):
      no_basin_track += 1

    # Update Fields...
    # Execute the Tracking Update!!!
'''

# Trackable / Updatable Layers!
# Note: The Tracking Quantities should be 
#discharge = soil.array("float", shape).fill(0.0)
#discharge_track = soil.array("float", shape).fill(0.0)

#momentum = soil.array("fvec2", shape).fill([0.0, 0.0])
#momentum_track = soil.array("fvec2", shape).fill([0.0, 0.0])

# Different Models for Constructing the Right
# We could also say that a model wants a given set of layers,
# and that the struct which it accepts is thereby fixed.
# The only thing that has to happen is that the struct has
# to be constructed. Subsequent "updates" and the erosion
# "update" code could be handled explicitly in python afterwards...

def main():

  '''
  We construct a model from the shape that we want.
  The model basically links all the data types together,
  so that we have a structure that lets us compute the
  various quantities that we are interested in.

  The model is passed to the particle, which can use it
  to get the various quantities that it needs.

  We should also make it so that the "tracking" property
  of the layers, i.e. the state estimator model, is later
  somehow elegantly integrated into the layer concept.

  This will require some specification of the model though.
  '''

  shape = soil.shape([512, 512])
  model = make_model(shape)

  print(model)

if __name__ == "__main__":
  main()