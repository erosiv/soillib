import soillib as soil

t = soil.tensor(soil.float32, soil.shape(8, 8))
print(t.elem)
print(t.host)
print(t.type)
print(t.size)
print(t.shape)
#t.gpu()
soil.set(t.buffer, 1)
print(t.host)
a = t.numpy()
print(a)

b = soil.tensor.from_numpy(a)
print(b.shape)