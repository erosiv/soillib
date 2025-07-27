#!/usr/bin/env python

from __common__ import *

import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    shape = image.shape
    print(f"File: {file}, {image.buffer.type}")
    normal = soil.normal(image.buffer, shape, image.meta.scale)
    data = normal.numpy(soil.shape(shape[0], shape[1], 3))

    data = 0.5 + 0.5*data
    plt.imshow(data)
    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  # data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40704_DGM_tif_Ebensee"
  #data = "merged.tiff"
  #data = "erosion_basic.tiff"
  data = "C:\\Users\\nicho\\Datasets\\test.tiff"


  main(data)