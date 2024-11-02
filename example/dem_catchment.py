#!/usr/bin/env python

'''
Catchment Area Computation w. Pysheds:
Also used to condition field for uniformity.
'''

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import soillib as soil

def plot_dem(model):

  grid, dem = model

  fig, ax = plt.subplots(figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
  plt.colorbar(label='Elevation (m)')
  plt.grid(zorder=0)
  plt.title('Digital elevation map', size=14)
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.tight_layout()
  plt.show()

def plot_flow(model):

  grid, fdir, dirmap = model

  fig = plt.figure(figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
  boundaries = ([0] + sorted(list(dirmap)))
  plt.colorbar(boundaries= boundaries,
              values=sorted(dirmap))
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.title('Flow direction grid', size=14)
  plt.grid(zorder=-1)
  plt.tight_layout()
  plt.show()

def plot_area(model):

  grid, acc = model

  fig, ax = plt.subplots(figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)
  im = ax.imshow(acc, zorder=2,
                cmap='cubehelix',
                norm=colors.LogNorm(1, acc.max()),
                interpolation='bilinear')
  plt.colorbar(im, ax=ax, label='Upstream Cells')
  plt.title('Flow Accumulation', size=14)
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.tight_layout()
  plt.show()

'''
Main Control Flow
'''

def main(filename):

  print(f"Loading DEM ({filename})...")

  grid = Grid.from_raster(filename)
  dem = grid.read_raster(filename)
  model = (grid, dem)

  print("Computing Flow...")

  dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
  flow = grid.flowdir(dem, dirmap=dirmap)

  print("Computing Catchment...")

  area = grid.accumulation(flow, dirmap=dirmap)
  plot_area((grid, area))

if __name__ == "__main__":
  #input = "/home/nickmcdonald/Datasets/HydroSHEDS/n40e010_con.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #input = "out_altmuenster.tiff"
  #input = "/home/nickmcdonald/Datasets/elevation.tiff"
  input = "/home/nickmcdonald/Datasets/elevation_conditioned.tiff"
  #input = "merge.tiff"
  #input = "conditioned.tiff"
  main(input)