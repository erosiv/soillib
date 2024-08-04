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
  print("type:", array.type)
  print("elem:", array.elem())
  print("size:", array.size())

  print("shape:", array.shape)
  array.reshape(soil.shape([3, 9]))
  print("shape:", array.shape)

  # Assignment
  array.zero()
  print(array[0])
  array.fill(3.14)
  print(array[0])
  array[0] = 1.

  # Iterate over Shape and Retrieve Values
  shape = array.shape
  for pos in shape.iter():
    val = array[pos]
    print(f"{pos}: {val}")

  for pos, val in array.iter():
    print(f"{pos}: {val}")

  # Numpy Interface
  numpy = np.array(array)
  print(numpy)
  print(numpy.dtype)

#test_shape()
test_array()