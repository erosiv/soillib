#!/usr/bin/env python

'''
DEM Conditioning Script
Make the Digital Elevation Model
Hydrologically Consistent for Drainage

Note: Currently using PySheds!
'''

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import soillib as soil
import rasterio
import math

'''
Main Control Flow
'''

def main(filename):

  print(f"Loading DEM ({filename})...")

  grid = Grid.from_raster(filename)
  dem = grid.read_raster(filename)

  print("Processing DEM...")

  dirmap = (7, 8, 1, 2, 3, 4, 5, 6)
  pour = (3733, 2640) # elevation_conditioned
  pour = (4670, 5951) # elevation_conditioned

#  pour = (1617, 973)  # gosau
#  pour = (1873, 692)  # bad goisern

  print("Computing Flow Index")
  with soil.timer() as timer:
    fdir = grid.flowdir(dem, dirmap=dirmap)
    
  print("Computing Area")
  with soil.timer() as timer:
    area = grid.accumulation(fdir, dirmap=dirmap)

  print("Computing Upstream Mask...")
  with soil.timer() as timer:
    catch = grid.catchment(x=pour[0], y=pour[1], fdir=fdir, dirmap=dirmap, 
      xytype='index')

  print("Computing Upstream Distance...")
  with soil.timer() as timer:
    dist = grid.distance_to_outlet(x=pour[0], y=pour[1], fdir=fdir, dirmap=dirmap,
      xytype='index')

  print("Plotting Data...")

  fig, ax = plt.subplots(2, 2, figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)

  im = ax[0, 0].imshow(fdir, zorder=2,
    cmap='CMRmap',
    interpolation='bilinear')

  im = ax[0, 1].imshow(area, zorder=2,
    cmap='CMRmap',
    norm=colors.LogNorm(1, area.max()),
    interpolation='bilinear')

  im = ax[1, 0].imshow(catch, zorder=2,
    cmap='CMRmap',
    interpolation='none')

  im = ax[1, 1].imshow(dist, zorder=2,
    cmap='CMRmap',
    interpolation='bilinear')

  plt.show()

if __name__ == "__main__":

  file_in = "/home/nickmcdonald/Datasets/elevation_conditioned.tiff"
  #file_in = "_dem_conditioned.tiff"

  main(file_in)