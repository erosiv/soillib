#!/usr/bin/env python

import os
import soillib as soil
import matplotlib.pyplot as plt

def main(path):

  path = os.fsencode(path)
  
  if not os.path.exists(path):
    raise RuntimeError("path does not exist")

  def show_tiff(path):

    img = soil.geotiff(path)
    buf = img.buf()
    data = buf.numpy()

    data = data.reshape((img.height, img.width))
    print(data.shape, data.dtype)

    plt.imshow(data)
    plt.show()

  if os.path.isfile(path):

    show_tiff(path)

  elif os.path.isdir(path):
    for file in os.listdir(path):
      show_tiff(os.path.join(path, file))

if __name__ == "__main__":

  data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen"
  main(data)