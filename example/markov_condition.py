#!/usr/bin/env python

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

'''
Markov-Chain Monte-Carlo DEM Conditioning
---
Basic Idea:
1. Flow Direction Grid is a Random State
2. Attempt to Lower Energy of State w. MCMC

Primary Problem:
1. Asymmetric Cost-Function Design
2. Efficient Implementation
3. Directed Conditioning of Height
'''

'''
dirmap = (6, 7, 0, 1, 2, 3, 4, 5)
dirset = [
  np.array([ 0, 1]),
  np.array([ 1, 1]),
  np.array([ 1, 0]),
  np.array([ 1,-1]),
  np.array([ 0,-1]),
  np.array([-1,-1]),
  np.array([-1, 0]),
  np.array([-1, 1])
]

def direction(code):
  return dirset[code]
'''

def compute_index(dem, flow):
  ind = np.indices(dem.shape).transpose((1,2,0))
  ind[flow ==   4] += np.array([ 1, 0])
  ind[flow ==  64] += np.array([-1, 0])

  ind[flow ==   2] += np.array([ 1, 1])
  ind[flow ==  32] += np.array([-1,-1])

  ind[flow ==   8] += np.array([ 1,-1])
  ind[flow == 128] += np.array([-1, 1])

  ind[flow ==   1] += np.array([ 0, 1])
  ind[flow ==  16] += np.array([ 0,-1])
  ind = np.clip(ind, (0,0), (dem.shape[0]-1, dem.shape[1]-1))
  return ind

def condition(model):

  # Note: Replace this Function!

  '''
  1. Cost of each local state
  2. Compute Cost for each transition (in theory)
  3. Sample Transitions
  4. Execute Transitions
  '''

  # Extract Initial Digital Elevation Model
  grid, dem = model

  # Compute the Flowmap
  dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
  flow = grid.flowdir(dem, dirmap=dirmap)

  # Initial Test: Basic Forward Propagate?

  '''
  Get the Flow Direction, Accumulation per Position
  -> Advance the Positions to the next position  
  '''

#   flow_next = flow[ind[..., 0], ind[..., 1]]
#   area_next = area[ind[..., 0], ind[..., 1]]
#   flow_step = flow.copy()

  # print("SHOWING PITS")
  # plt.imshow(flow < 0)
  # plt.show()

  def opposite(flow, x, y, i):
    if flow[i[0],i[1]] ==   1 and flow[x,y] ==  16: return True
    if flow[i[0],i[1]] ==   2 and flow[x,y] ==  32: return True
    if flow[i[0],i[1]] ==   4 and flow[x,y] ==  64: return True
    if flow[i[0],i[1]] ==   8 and flow[x,y] == 128: return True
    if flow[i[0],i[1]] ==  16 and flow[x,y] ==   1: return True
    if flow[i[0],i[1]] ==  32 and flow[x,y] ==   2: return True
    if flow[i[0],i[1]] ==  64 and flow[x,y] ==   4: return True
    if flow[i[0],i[1]] == 128 and flow[x,y] ==   8: return True
    return False

  # Do it energy based... makes thing simpler and more obvious
  # what stable configurations are!
  def energy(flow, x, y, i):

    # How do we do it?

    return 0.0

  for n in range(64):

    print("Invalid:", np.sum(flow <= 0))
    if(np.sum(flow <= 0) == 0):
      break

    ind = compute_index(dem, flow)

    area = grid.accumulation(flow, dirmap=dirmap)

    # Note: Remove Iteration Ordering Bias

    flow_next = flow.copy()

    print(2**16)
    for k in range(2**16):

      # Sample Random Position
      p = np.random.rand(2)
      x = (dem.shape[0] * p[0]).astype(np.int64)
      y = (dem.shape[1] * p[1]).astype(np.int64)
      i = ind[x,y]

      # Mutate
      # Question: What is the op

      if flow[i[0],i[1]] == -1:
        flow_next[i[0], i[1]] = flow[x,y]

      if flow[i[0],i[1]] == -2:
        flow_next[i[0], i[1]] = flow[x,y]

#      Correct Opposite Directions?

      if opposite(flow, x, y, i):
        if area[i[0],i[1]] > area[x,y]:
          #Next is smaller: upstream
          flow_next[x,y] = flow[i[0], i[1]]
        else:
          flow_next[i[0], i[1]] = flow[x,y]

    flow = flow_next

  print("Invalid: ", np.sum(flow < 0))
  return (grid, dem, dirmap, flow)

'''
Rest of the Model
'''

# Load the Digital Elevation Model
def load(filename):

  grid = Grid.from_raster(filename)
  dem = grid.read_raster(filename)

  return (grid, dem)


def flow(model):

  grid, dem = model

  dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
  fdir = grid.flowdir(dem, dirmap=dirmap)

  return (grid, dem, dirmap, fdir)

def catchment(model):

  grid, dem, dirmap, fdir = model

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

  grid, dem, dirmap, fdir = model

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
  acc = np.transpose(acc)

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

  print("Loading File...")
  model = load(filename)
  
  print("Conditioning DEM...")
  fmodel = condition(model)
  #fmodel = flow(model)

  #print("Plotting DEM...")
  #plot_dem(model)

  # print("Computing Flow...")

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
  #input = "out.tiff"
  main(input)