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

  pour = (3733, 2640) # elevation_conditioned
  pour = (4670, 5951) # elevation_conditioned
#  pour = (1617, 973)  # gosau
#  pour = (1873, 692)  # bad goisern

  print("Computing Flow Index")
  with soil.timer() as timer:
    flow = soil.flow(buffer, index)

  print("Computing Direction")
  with soil.timer() as timer:
    direction = soil.direction(flow, index)
  
  print("Computing Area")
  with soil.timer() as timer:
    area = soil.accumulation(flow, index, 2*16, 2*2048, 8192)

  print("Computing Upstream Mask...")
  with soil.timer() as timer:
    catch = soil.upstream(direction, index, [pour[1], pour[0]])

  print("Computing Upstream Distance...")
  with soil.timer() as timer:
    distance = soil.distance(direction, index, [pour[1], pour[0]])

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

  data = "/home/nickmcdonald/Datasets/elevation_conditioned.tiff"
  #data = "_dem_conditioned.tiff"

  main(data)