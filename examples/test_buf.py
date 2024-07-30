#!/usr/bin/env python

import soillib as soil

# print(shape.elem())
# print(shape[0])
# print(shape[1])

shape = soil.shape([64, 64])
print(shape.elem())

'''
#print(shape[1])
for d in shape:
  print(d)
  print(d[0])
  #print(d[0], d[1])
  #print(d.flat(shape))
'''


'''
buf = soil.buffer("float", 32).to("float")
print(buf.elem())
print(buf.size())

buf.fill(3.14)
numpy = buf.numpy()
print(numpy)
print(numpy.dtype)
'''