#!/usr/bin/env python

import soillib as soil
import numpy as np
import time

def test_layer():

  '''
  height =      soil.array(soil.float32, shape).fill(0.0)  
  discharge =   soil.array(soil.float32, shape).fill(0.0)
  momentum =    soil.array(soil.vec2, shape).fill([0.0, 0.0])

  resistance =  soil.constant(soil.float32, 0.0)
  maxdiff =     soil.constant(soil.float32, 0.8)
  settling =    soil.constant(soil.float32, 1.0)
  '''

  x = 1.0
  j = 0.0
  print("Computed Value:")
  with soil.timer() as timer:
    comp = soil.computed(soil.float32, lambda index: x)
    for i in range(2**18):
      j = comp(0)
  print(j)

  j = 0.0
  print("Constant Value:")
  with soil.timer() as timer:
    const = soil.constant(soil.float32, 1.0)
    for i in range(2**18):
      j = const(0)
  print(j)

test_layer()