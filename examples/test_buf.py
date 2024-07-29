#!/usr/bin/env python

import soillib as soil

buf = soil.buffer("float", 32).to("float")
print(buf.elem())
print(buf.size())

buf.fill(3.14)
numpy = buf.numpy()
print(numpy)
print(numpy.dtype)