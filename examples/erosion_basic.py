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
  resistance = soil.constant("float", 0.0)
  
  #print(momentum(0))
  
  #discharge = soil.layer(lambda d: erf(d))

  return soil.water_model(
    shape,
    height,
    momentum,
    discharge,
    resistance
  )

def erode(model, n_cycles = 512):

  no_basin_track = 0.0

  for n in range(n_cycles):

    print(f"Erosion Step ({n})")

    pos = 512*np.random.rand(2)
    drop = soil.water(pos)
    print(drop.pos)

    steps = 0
    while(True):

      if not drop.move(model):
        break

      # ... update the tracking maps ...
      # discharge_track[particle.pos] += particle.volume
      # momentum_track[particle.pos] += particle.volume * particle.speed

      if not drop.interact(model):
        break

      steps += 1

    print(steps)
    print(drop.pos)

    '''
    if oob(particle.pos):
      no_basin_track += 1
    '''

    # Update Fields...
    # Execute the Tracking Update!!!

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


  '''
  Construct Initial Condition
  '''




  erode(model, n_cycles = 512)

  #print(model)
  #print(model.height[0])

if __name__ == "__main__":
  main()