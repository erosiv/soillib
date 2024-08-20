#!/usr/bin/env python

import soillib as soil
import numpy as np
import time

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

def test_buffer():

  '''
  define and test the soil.buffer type interface
  '''

  elem = 4096

  # Int Buffer

  print(f"Testing soil.buffer({soil.int})...")

  buffer = soil.buffer(soil.int, elem)
  assert buffer.type == soil.int
  assert buffer.elem() == elem
  assert buffer.size() == 4*elem

  buffer.zero()
  assert buffer[0] == 0

  values = [6, 9]

  buffer.fill(values[0])
  assert np.isclose(buffer[0], values[0])

  buffer[0] = values[1]
  assert np.isclose(buffer[0], values[1])
  assert np.isclose(buffer[1], values[0])

  numpy = buffer.numpy()
  assert np.isclose(numpy[0], values[1])
  assert np.isclose(numpy[1], values[0])

  # Float32 Buffer

  print(f"Testing soil.buffer({soil.float32})...")

  buffer = soil.buffer(soil.float32, elem)
  assert buffer.type == soil.float32
  assert buffer.elem() == elem
  assert buffer.size() == 4*elem

  buffer.zero()
  assert buffer[0] == 0

  values = [3.14, 1.9]

  buffer.fill(values[0])
  assert np.isclose(buffer[0], values[0])

  buffer[0] = values[1]
  assert np.isclose(buffer[0], values[1])
  assert np.isclose(buffer[1], values[0])

  numpy = buffer.numpy()
  assert np.isclose(numpy[0], values[1])
  assert np.isclose(numpy[1], values[0])

  # Float64 Buffer

  print(f"Testing soil.buffer({soil.float64})...")

  buffer = soil.buffer(soil.float64, elem)
  assert buffer.type == soil.float64
  assert buffer.elem() == elem
  assert buffer.size() == 8*elem

  buffer.zero()
  assert buffer[0] == 0

  values = [3.14, 1.9]

  buffer.fill(values[0])
  assert np.isclose(buffer[0], values[0])

  buffer[0] = values[1]
  assert np.isclose(buffer[0], values[1])
  assert np.isclose(buffer[1], values[0])

  numpy = buffer.numpy()
  assert np.isclose(numpy[0], values[1])
  assert np.isclose(numpy[1], values[0])

  # Vec2 Buffer

  print(f"Testing soil.buffer({soil.vec2})...")

  buffer = soil.buffer(soil.vec2, elem)
  assert buffer.type == soil.vec2
  assert buffer.elem() == elem
  assert buffer.size() == 8*elem

  buffer.zero()
  assert buffer[0] == [0, 0]

  values = [[3.14, 1.9], [2.0, 1.5]]

  buffer.fill(values[0])
  assert np.isclose(buffer[0], values[0]).all()

  buffer[0] = values[1]
  assert np.isclose(buffer[0], values[1]).all()
  assert np.isclose(buffer[1], values[0]).all()

  numpy = buffer.numpy()

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
test_buffer()