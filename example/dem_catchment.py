#!/usr/bin/env python

'''
Catchment Area Computation w. Pysheds:
Also used to condition field for uniformity.
'''

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
import soillib as soil
from __common__ import *

'''
Main Control Flow
'''

dirmap = (7, 8, 1, 2, 3, 4, 5, 6)
coords = [
  np.array([ 0,-1]),  # N
  np.array([ 1,-1]),  # NE
  np.array([ 1, 0]),  # E
  np.array([ 1, 1]),  # SE
  np.array([ 0, 1]),  # S
  np.array([-1, 1]),  # SW
  np.array([-1, 0]),  # W
  np.array([-1,-1]),  # NW
]

def calc_d8(data):

  slope_stack = np.full((8, data.shape[0], data.shape[1]), 0.0)
  for k, coord in enumerate(coords):
    dist = np.sqrt(coord[0]**2 + coord[1]**2)
    slope_stack[k] = (data_copy - np.roll(data, (-coord[0], -coord[1]), axis=(1,0)))/dist

  d8 = np.argmax(slope_stack, axis=0)
  d8 = np.asarray(list(dirmap))[d8]
  return d8

def main(filename):

  print(f"Loading DEM ({filename})...")

  grid = Grid.from_raster(filename)
  dem = grid.read_raster(filename)

  print("Computing Flow...")

  flow = grid.flowdir(dem, dirmap=dirmap)
#  my_flow = calc_d8(np.copy(dem))
#  flow[1:-2,1:-2] = my_flow[1:-2,1:-2]

  print("Computing Catchment...")

  area = grid.accumulation(flow, dirmap=dirmap)

  #catch = grid.catchment(x=1873, y=692, fdir=flow, dirmap=dirmap, xytype='index')
  #area[catch == 0] = 1
  plot_area(area)

if __name__ == "__main__":
  #input = "/home/nickmcdonald/Datasets/HydroSHEDS/n40e010_con.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #input = "out_altmuenster.tiff"
  #input = "/home/nickmcdonald/Datasets/elevation.tiff"
  input = "/home/nickmcdonald/Datasets/elevation_conditioned.tiff"
  input = "merge.tiff"
  input = "conditioned.tiff"
  main(input)

  #1640, 812