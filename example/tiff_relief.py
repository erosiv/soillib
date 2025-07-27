#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")

    shape = image.shape
    height = image.buffer.numpy(shape)
    normal = soil.normal(image.buffer, shape, image.meta.scale).numpy(soil.shape(shape[0], shape[1], 3))

    a = soil.buffer(soil.float32, 8).gpu()
    b = soil.buffer(soil.float32, 8).gpu()
    soil.set(a, 1)
    soil.set(b, 6)

    soil.add(a, b)
    a = a.cpu().numpy(soil.shape(2, 4))
    print(a)

    # Compute Shading
    relief = relief_shade(height, normal)
    plt.imshow(relief, cmap='gray')
    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
#  data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40704_DGM_tif_Ebensee"
  data = "C:\\Users\\nicho\\Datasets\\test.tiff"

  main(data)