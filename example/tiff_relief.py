#!/usr/bin/env python

from __common__ import *

import soillib as soil
import numpy as np

def main(input):

  for file, path in iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.buffer.type}")

    index = image.index
    height_node = soil.cached(image.buffer)
    normal_node = soil.normal(index, height_node)

    height_data = height_node.buffer.numpy(index)
    normal_data = soil.bake(normal_node, index).numpy(index)

    # Compute Shading
    relief = relief_shade(height_data, normal_data)
    plt.imshow(relief, cmap='gray')
    plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Datasets/ViennaDGM/21_Floridsdorf"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  data = "merge.tiff"
  main(data)