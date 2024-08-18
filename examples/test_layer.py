#!/usr/bin/env python

import soillib as soil
import numpy as np
import time

def test_layer():

  print("Constant Value:")
  j = 0.0
  with soil.timer() as timer:
    const = soil.constant(soil.float32, 1.0)
    for i in range(2**18):
      j = const(0)
  print(j)

  print("Computed Value:")
  x = 1.0
  j = 0.0
  with soil.timer() as timer:
    comp = soil.computed(soil.float32, lambda i: 0.0)
    for i in range(2**18):
      j = comp(0)
  print(j)

  print("Coupled Layers:")
  with soil.timer() as timer:
    const1 = soil.constant(soil.float32, 3.14)
    const2 = soil.constant(soil.float32, 3.14)
    comp = soil.computed(soil.float32, lambda index:
      const1(index) + const2(index))
    for i in range(2**18):
      j = comp(0)
  print(j)

test_layer()