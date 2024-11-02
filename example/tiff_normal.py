#!/usr/bin/env python

from __common__ import *

import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    node = image.node()

    print(f"File: {file}, {node.type}")

    normal = soil.normal(image.index, node)
    data = normal.full().numpy(image.index)
    data = 0.5 + 0.5*data
    plt.imshow(data)
    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "out_ebensee.tiff"
  data = "/home/nickmcdonald/Datasets/elevation.tiff"

  main(data)