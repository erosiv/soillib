#!/usr/bin/env python

import soillib as soil

buf = soil.buffer("float", 32)
buf = buf.ast("float")

buf.fill(3.14)
print(buf.size())
print(buf.elem())

teset = buf.numpy()
print(teset.dtype)
print(teset)

print(buf[0])
print(buf[1])
print(buf[31])