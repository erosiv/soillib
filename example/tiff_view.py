#!/usr/bin/env python

import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def main(input):

  for file, path in soil.util.iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.tensor.type}")

    data = image.tensor.numpy()
    plt.imshow(data)
    plt.show()

if __name__ == "__main__":
  data = "data/dem_1024.tiff"
  main(data)