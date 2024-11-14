#!/usr/bin/env python

from __common__ import *

import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")

    mesh = soil.mesh(image.buffer, image.index)
    mesh.write("out.ply")

#    data = image.buffer.numpy(image.index)
#    plt.imshow(data)
#    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/HydroSHEDS/n40e010_con.tif"
  #data = "/home/nickmcdonald/Datasets/elevation.tiff"
  #data = "./conditioned.tiff"
  data = "merge.tiff"

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"

  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/41225_DGM_tif_Ried"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/41234_DGM_tif_Waldzell"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/41403_DGM_tif_Brunnenthal"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/41709_DGM_tif_Frankenburg"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40706_DGM_tif_Gosau"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40702_DGM_tif_Bad_Goisern"

  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40704_DGM_tif_Ebensee"

  main(data)
