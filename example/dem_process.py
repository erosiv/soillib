#!/usr/bin/env python

import soillib as soil
import matplotlib.pyplot as plt
from matplotlib import colors

'''
Example Code Showcasing Various
DEM Processing Kernels
'''

def load(data):

  image = soil.geotiff(data)
  print(f"Loaded File: {data}, {image.buffer.type}, ({image.index[0]}, {image.index[1]})")
  return (image.buffer, image.index)

def main(data):

  print("Loading Data")
  buffer, index = load(data)
  buffer.gpu()

  print("Computing Flow Index")
  with soil.timer() as timer:
    flow = soil.flow(buffer, index)

  print("Computing Direction")
  with soil.timer() as timer:
    direction = soil.direction(flow, index)
  
  print("Computing Area")
  with soil.timer() as timer:
    area = soil.accumulation(flow, index, 16, 4*4096, 8192)

  print("Computing Upstream Mask...")
  with soil.timer() as timer:
    catch = soil.upstream(direction, index, [692, 1873])
#    catch = soil.upstream(direction, index, [2640, 3733])

  print("Computing Upstream Distance...")
  with soil.timer() as timer:
    distance = soil.distance(direction, index, [692, 1873])
#    distance = soil.distance(direction, index, [2640, 3733])

  # Extract to Numpy
  area = area.cpu().numpy(index)
  catch = catch.cpu().numpy(index)
  distance = distance.cpu().numpy(index)
  flow = flow.cpu().numpy(index)

  print("Plotting Data...")

  fig, ax = plt.subplots(2, 2, figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)

  im = ax[0, 0].imshow(flow, zorder=2,
    cmap='CMRmap',
    interpolation='bilinear')

  im = ax[0, 1].imshow(area, zorder=2,
    cmap='CMRmap',
    norm=colors.LogNorm(1, area.max()),
    interpolation='bilinear')

  im = ax[1, 0].imshow(catch, zorder=2,
    cmap='CMRmap',
    interpolation='none')

  im = ax[1, 1].imshow(distance, zorder=2,
    cmap='CMRmap',
    interpolation='bilinear')

  plt.show()

if __name__ == "__main__":

  #data = "/home/nickmcdonald/Downloads/elevation.tiff"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #data = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #data = "out_altmuenster.tiff"
  #data = "/home/nickmcdonald/Datasets/elevation_conditioned.tiff"
  data = "conditioned.tiff"
  #data = "erosion_basic.tiff"

  main(data)