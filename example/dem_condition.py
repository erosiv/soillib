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

  print("Conditioning DEM...")

  dem.nodata = np.nan
  dem = grid.fill_pits(dem)
  dem = grid.fill_depressions(dem)
  dem = grid.resolve_flats(dem)

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