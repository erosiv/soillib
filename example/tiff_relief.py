#!/usr/bin/env python

import soillib as soil
import silt
import matplotlib.pyplot as plt

def main(input):

  for file, path in soil.util.iter_tiff(input):

    image = soil.geotiff(path)
    print(f"File: {file}, {image.tensor.type}")

    # Compute Shading
    # soil.normal runs on the GPU; image.tensor loads CPU-resident.
    height = image.tensor.numpy()
    height_gpu = image.tensor.copy_to(silt.gpu)
    normal = soil.normal(height_gpu, image.meta.scale).copy_to(silt.cpu).numpy()
    relief = soil.util.relief_shade(height, normal)
    plt.imshow(relief, cmap='gray')
    plt.show()

if __name__ == "__main__":
  data = "data/dem_1024.tiff"
  main(data)
