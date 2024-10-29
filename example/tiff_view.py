#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt
import numpy as np

def iter_tiff(path):

  '''
  Generator for all Files in 
  Directory, or a Single File
  '''

  path = os.fsencode(path)
  if not os.path.exists(path):
    raise RuntimeError("path does not exist")

  if os.path.isfile(path):
    file = os.path.basename(path)
    yield file, path.decode('utf-8')

  elif os.path.isdir(path):
    for file in os.listdir(path):
      yield file, os.path.join(path, file).decode('utf-8')

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
  data = "./out.tiff"
  main(data)