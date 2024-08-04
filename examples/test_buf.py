#!/usr/bin/env python

import soillib as soil

def test_shape():

  shape = soil.shape([4, 4])
  print(shape)
  print(shape.elem())

  for t in shape.iter():
    print(t)

  # print(shape.elem())
  # print(shape[0])
  # print(shape[1])

  '''
  #print(shape[1])
  for d in shape:
    print(d)
    print(d[0])
    #print(d[0], d[1])
    #print(d.flat(shape))
  '''

def test_buf():

  buf = soil.array("float", [4, 4])#.to("float")
  print(buf.shape())
  print(buf.type())
  print(buf.elem())
  print(buf.size())

  buf.zero()

  print(buf[0])

  # for test in buf.shape().iter():
  #   print(test)

  buf.fill(3.14)


  print(buf[0])
  buf[0] = 1.

  numpy = buf.numpy()
  print(numpy)
  print(numpy.dtype)

test_shape()
test_buf()