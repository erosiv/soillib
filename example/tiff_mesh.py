#!/usr/bin/env python

import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def main(input, file_out):

  for file, path in soil.util.iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.tensor.type}")

    scale = image.scale
    mesh = soil.mesh(image.tensor, [scale[0], scale[1], 1])
    mesh.center()
    mesh.write_binary(file_out)

if __name__ == "__main__":
  data = "data/dem_1024.tiff"
  file_out = "data/mesh.ply"
  main(data, file_out)