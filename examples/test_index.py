#!/usr/bin/env python

import soillib as soil
import numpy as np
import time

def test_shape():

  '''
  define and test the soil.shape type interface
  '''

  # Construction / General Properties
  shape = soil.shape([3, 9])
  print("shape:", shape)
  print("dims:", shape.dims())
  print("elem:", shape.elem())

  # Print the Dimension Extents
  for d in range(shape.dims()):
    print(f"d:{d}, e:{shape[d]}")

  # Iterate over Dimension Extents
  for pos in shape.iter():
    print(f"i: {shape.flat(pos)}: {pos}")

def test_iter():

  print("Empty Python Range:")
  with soil.timer() as timer:
    for x in range(2048):
      for y in range(2048):
        pos = [x, y]
        #index = x * 2048 + y
        #print(type(pos))
        #i = i+1
        pass

  print("Shape Generator Iterator:")
  shape = soil.shape([2048, 2048])
  with soil.timer() as timer:
    for pos in shape.iter():
      #index = shape.flat(pos)
      #print(type(pos))
      pass

  arr = np.empty((2048, 2048))
  print("Numpy Shape Iterator:")
  with soil.timer() as timer:
    for x in arr:
      for y in x:
        #t = [x, y]
        #print(x, y)
        pass

  '''
  print("Shape Classic Iterator:")
  with soil.timer() as timer:
    for pos in shape:
      #rint(pos)
      #index = shape.flat(pos)
      #print(type(pos))
      pass
  '''

test_shape()
test_iter()
