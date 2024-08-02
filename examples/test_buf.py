#!/usr/bin/env python

import soillib as soil

# print(shape.elem())
# print(shape[0])
# print(shape[1])

'''
shape = soil.shape([4, 4])

print(shape)
print(shape.elem())

for t in shape.iter():
  print(t)
'''



'''
#print(shape[1])
for d in shape:
  print(d)
  print(d[0])
  #print(d[0], d[1])
  #print(d.flat(shape))
'''


buf = soil.buffer("float", [4, 4]).to("float")
print(buf.shape())
print(buf.shape())
print(buf.elem())
print(buf.size())

for test in buf.shape().iter():
  print(test)

buf.fill(3.14)
numpy = buf.numpy()
print(numpy)
print(numpy.dtype)