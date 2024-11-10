#!/usr/bin/env python

import soillib as soil
import numpy as np
import time

print("Testing layer.constant...")

value = 3.14
layer = soil.constant(soil.float32, value)
assert type(layer) == soil._node
assert layer.type == soil.float32
assert vars(layer) == {}

for i in range(64):
  assert np.isclose(layer(0), value)

print("Testing layer.cached...")

index = soil.index([32, 32])
value = [0.0, 3.14]

#buffer = soil.buffer(soil.vec2, index.elem())
#layer = soil.cached(buffer)
layer = soil.cached(soil.vec2, index.elem())
assert type(layer) == soil._node
assert layer.type == soil.vec2
assert layer.buffer != None

soil.set(layer.buffer, value)
for i in range(index.elem()):
  assert np.isclose(layer(i)[0], value[0])
  assert np.isclose(layer(i)[1], value[1])

print("Testing layer.computed...")

layer = soil.computed(soil.int, lambda i: i)
assert type(layer) == soil._node
assert layer.type == soil.int
assert vars(layer) == {}

for i in range(64):
  assert layer(i) == i

'''
print("Testing coupled layer.computed")

with soil.timer() as timer:
  const1 = soil.constant(soil.float32, 3.14)
  const2 = soil.constant(soil.float32, 3.14)
  comp = soil.computed(soil.float32, lambda index:
    const1(index) + const2(index))
  for i in range(2**18):
    j = comp(0)
print(j)
'''