#!/usr/bin/env python

'''
Catchment Area Computation w. Pysheds:
Also used to condition field for uniformity.
'''

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Load the Digital Elevation Model
def load(filename):

  grid = Grid.from_raster(filename)
  dem = grid.read_raster(filename)

  return (grid, dem)

def condition(model):

  grid, dem = model

  dem_pit = grid.fill_pits(dem)               # Fill Single Pits
  dem_flood = grid.fill_depressions(dem_pit)  # Fill Large Pits
  dem_slope = grid.resolve_flats(dem_flood)   # Fix Flat Sections

  return (grid, dem_slope)

def flow(model):

  grid, dem = model

  dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
  fdir = grid.flowdir(dem, dirmap=dirmap)

  return (grid, fdir, dirmap)

def catchment(model):

  grid, fdir, dirmap = model

  acc = grid.accumulation(fdir, dirmap=dirmap)
 
  return (grid, acc)

'''
Plotting Functions
'''

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

def plot_acc(model):

  grid, acc = model

  fig, ax = plt.subplots(figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)
  im = ax.imshow(acc, extent=grid.extent, zorder=2,
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

  print("Loading File...")
  model = load(filename)
  
  print("Conditioning DEM...")
  model = condition(model)

  #print("Plotting DEM...")
  #plot_dem(model)

  print("Computing Flow...")
  fmodel = flow(model)

  #print("Plotting Flow...")
  #plot_flow(fmodel)

  print("Computing Catchment...")
  amodel = catchment(fmodel)
  plot_acc(amodel)

if __name__ == "__main__":

  #input = "/home/nickmcdonald/Downloads/elevation.tiff"
  input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #input = "out_altmuenster.tiff"

  main(input)