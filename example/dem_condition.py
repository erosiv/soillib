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

'''
Main Control Flow
'''

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

def main(filename):

  print(f"Loading DEM ({filename})...")

  grid = Grid.from_raster(filename)
  dem = grid.read_raster(filename)

  print("Conditioning DEM...")

  dem = grid.fill_pits(dem)
  dem = grid.fill_depressions(dem)
  dem = grid.resolve_flats(dem)

  print("Saving DEM...")

  # Note: Make this construction easier!
  # Note: This has to be double precision,
  # because the correction algorithms use double precision.
  # If you don't do this, accuracy is lost and the state
  # is not accurately reproduced for flow direction purposes.

  shape = soil.index(dem.shape)
  array = soil.buffer(soil.float64, shape.elem())
  array.fill(np.nan)

  for x in range(shape[0]):
    for y in range(shape[1]):
      array[x*shape[1]+y] = dem[x,y]

  t = soil.geotiff()
  t.meta(filename)

  tiff_out = soil.geotiff(array, shape)
  tiff_out.set_meta(t.get_meta())
  tiff_out.unsetnan()
  tiff_out.write("conditioned.tiff")

  print("Computing Flow...")

  dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
  flow = grid.flowdir(dem, dirmap=dirmap)

  print("Computing Catchment...")

  area = grid.accumulation(flow, dirmap=dirmap)
  plot_area((grid, area))

if __name__ == "__main__":

  #input = "/home/nickmcdonald/Datasets/elevation.tiff"
  #input = "/home/nickmcdonald/Datasets/HydroSHEDS/n40e010_con.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"

  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40704_DGM_tif_Ebensee/G-T4830-22.tif"
  #input = "out_altmuenster.tiff"
  #input = "out_cond.tiff"
  input = "merge.tiff"
  main(input)