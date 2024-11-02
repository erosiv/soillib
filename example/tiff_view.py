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

    data = node.numpy(image.index)
    plt.imshow(data)
    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/HydroSHEDS/n40e010_con.tif"
  #data = "/home/nickmcdonald/Datasets/elevation.tiff"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40704_DGM_tif_Ebensee"
  #data = "./elevation_conditioned.tiff"
  data = "merge.tiff"

  main(data)