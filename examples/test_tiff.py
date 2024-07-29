#!/usr/bin/env python

import soillib as soil
import numpy as np
import matplotlib.pyplot as plt

data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-70.tif"
# data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf/15_2_dgm.tif"

img = soil.geotiff(data)
buf = img.buf()

data = buf.numpy()
data = data.reshape((img.height, img.width))
print(data.shape, data.dtype)

plt.imshow(data)
plt.show()