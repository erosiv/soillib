#!/usr/bin/env python

import soillib as soil
import numpy as np
import time

'''
Test the Index Types
- Construction, Return Values
- Iteration Schemes
- Numpy Export
'''

print(f"Testing soil.index({soil.flat1})...")

array = [32]
shape = soil.index(array)

assert shape.type() == soil.flat1
assert shape.dims() == 1
assert shape.elem() == array[0]

for d in range(shape.dims()):
  assert shape[d] == array[d]

assert not shape.oob(shape.min())
assert shape.oob(shape.max())

i = 0
for pos in shape.iter():
  assert not shape.oob(pos)
  assert shape.flatten(pos) == i
  i += 1
assert i == shape.elem()

print(f"Testing soil.index({soil.flat2})...")

array = [16, 32]
shape = soil.index(array)

assert shape.type() == soil.flat2
assert shape.dims() == 2
assert shape.elem() == array[0]*array[1]

for d in range(shape.dims()):
  assert shape[d] == array[d]

assert not shape.oob(shape.min())
assert shape.oob(shape.max())

i = 0
for pos in shape.iter():
  assert not shape.oob(pos)
  assert shape.flatten(pos) == i
  i += 1
assert i == shape.elem()

print(f"Testing soil.index({soil.flat3})...")

array = [8, 16, 32]
shape = soil.index(array)

assert shape.type() == soil.flat3
assert shape.dims() == 3
assert shape.elem() == array[0]*array[1]*array[2]

for d in range(shape.dims()):
  assert shape[d] == array[d]

assert not shape.oob(shape.min())
assert shape.oob(shape.max())

i = 0
for pos in shape.iter():
  assert not shape.oob(pos)
  assert shape.flatten(pos) == i
  i += 1
assert i == shape.elem()

print(f"Testing soil.index({soil.flat4})...")

array = [4, 8, 16, 32]
shape = soil.index(array)

assert shape.type() == soil.flat4
assert shape.dims() == 4
assert shape.elem() == array[0]*array[1]*array[2]*array[3]

for d in range(shape.dims()):
  assert shape[d] == array[d]

assert not shape.oob(shape.min())
assert shape.oob(shape.max())

i = 0
for pos in shape.iter():
  assert not shape.oob(pos)
  assert shape.flatten(pos) == i
  i += 1
assert i == shape.elem()

print(f"Testing soil.index({soil.quad})...")

array = [
  ([ 0,  0], [64, 64]), # Min, Extent
  ([64, 32], [32, 32]), # Min, Extent
  ([80, 16], [16, 16]), # Min, Extent
  ([72, 16], [ 8,  8]), # Min, Extent
  ([72, 24], [ 4,  4]), # Min, Extent
]

shape = soil.index(array)

assert shape.type() == soil.quad
assert shape.dims() == 2
assert shape.elem() == 64*64 + 32*32 + 16*16 + 8*8 + 4*4

i = 0
for pos in shape.iter():
  assert not shape.oob(pos)
  assert shape.flatten(pos) == i
  i += 1
assert i == shape.elem()