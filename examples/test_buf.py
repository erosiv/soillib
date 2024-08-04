#!/usr/bin/env python

import soillib as soil
import numpy as np

def test_shape():

  '''
  define and test the soil.shape type interface
  '''

  # Construction / General Properties
  shape = soil.shape([3, 3, 3])
  print("shape:", shape)
  print("dims:", shape.dims())
  print("elem:", shape.elem())

  # Print the Dimension Extents
  for d in range(shape.dims()):
    print(f"d:{d}, e:{shape[d]}")

  # Iterate over Dimension Extents
  for pos in shape.iter():
    print(f"i: {shape.flat(pos)}: {pos}")

def test_array():

  '''
  define and test the soil.array type interface
  '''

  # Construction / General Properties
  array = soil.array("float", [3, 3, 3])
  print("shape:", array.shape)
  print("type:", array.type)
  print("elem:", array.elem())
  print("size:", array.size())

  # Assignment
  array.zero()
  print(array[0])
  array.fill(3.14)
  print(array[0])
  array[0] = 1.

  # Numpy Interface
  numpy = np.array(array)
  
  #numpy = array.numpy()
  print(numpy)
  print(numpy.dtype)

#test_shape()
test_array()