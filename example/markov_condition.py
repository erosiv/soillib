#!/usr/bin/env python

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm

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

# Flow Direction Set and Lookup

'''
 6  7  8
 5  0  1
 4  3  2
'''

dirmap = (7, 8, 1, 2, 3, 4, 5, 6)
dirset = np.array([
  np.array([ 0, 1]),
  np.array([ 1, 1]),
  np.array([ 1, 0]),
  np.array([ 1,-1]),
  np.array([ 0,-1]),
  np.array([-1,-1]),
  np.array([-1, 0]),
  np.array([-1, 1])
])

'''
Energy Function:
Option 1:
-> Simple Neighborhood Majority

Option 2:
-> Incoming Vector Weighting

Option 3:
-> Something w. Divergence
-> Divergence can't vanish!
'''

coords = [
  [0, 0],
  [1, 0],
  [2, 0],
  [0, 1],
  [2, 1],
  [0, 2],
  [1, 2],
  [2, 2]
]

# Cost Matrix!
cost = np.array([
  [0, 1, 2, 3, 4, 3, 2, 1],
  [1, 0, 1, 2, 3, 4, 3, 2],
  [2, 1, 0, 1, 2, 3, 4, 3],
  [3, 2, 1, 0, 1, 2, 3, 4],
  [4, 3, 2, 1, 0, 1, 2, 3],
  [3, 4, 3, 2, 1, 0, 1, 2],
  [2, 3, 4, 3, 2, 1, 0, 1],
  [1, 2, 3, 4, 3, 2, 1, 0]
])

def energy(state, area, i):

  '''
  Energy Function:
  - Inward Pointing Neighbors Contribute
  - Angle Determines Cost!
  - Same Direction = 0
  - Todo: Weight by Catchment Area
  
  Note: We can probably motivate the energy function
    simply using the Cauchy-Momentum Equation.
  '''

  e = 0.0
  t = dirmap[i]-1
  for coord in coords:
    s = state[coord[0], coord[1]]-1
    a = area[coord[0], coord[1]]
    if s < 0: continue  # Ignore Negative Values on the perimeter?
    dir = dirset[s]
    if coord[0] + dir[0] != 1: continue
    if coord[1] + dir[1] != 1: continue
    c = cost[s, t]
    e += a*c*c  # Note: Square! (otherwise directionally biased)
  return e

def condition(model):

  # Note: Replace this Function!

  '''
  1. Cost of each local state
  2. Compute Cost for each transition (in theory)
  3. Sample Transitions
  4. Execute Transitions
  '''

  # Extract Initial Digital Elevation Model
  grid, dem = model                               # DEM, Grid
  flow = grid.flowdir(dem, dirmap=dirmap)         # Flow from DEM
  area = grid.accumulation(flow, dirmap=dirmap)
  ext = np.array([dem.shape[0], dem.shape[1]])

  # First Step: Correct Pits?

  for n in range(1):

    print("Invalid:", np.sum(flow < 0))
    for x in range(ext[0]-2):
      x += 1
      for y in range(ext[1]-2):
        y += 1

        if flow[x, y] > 0:
          continue
        
        f_state = flow[x-1:x+2, y-1:y+2]
        a_state = area[x-1:x+2, y-1:y+2]

        E = []
        for i in range(8):
          E = np.append(E, energy(f_state, a_state, i))

        t = np.argmin(E)
        flow[x, y] = dirmap[t]

  N = 64     # Number of Iterations
  K = 2**14 # Samples per Iteration

  for n in range(N):

    print(n)

#    flow_next = flow.copy()
    area = grid.accumulation(flow, dirmap=dirmap)

    # Note: Add a way to compute total energy cost

    for k in tqdm(range(K)):

      # 1. Sample Random Position (Ignore Boundary)
      p = 1 + (np.random.rand(2)*(ext-2)).astype(np.int64)
      x = p[0]
      y = p[1]

      # 2. Get Local State, Compute Energy Vector

      f_state = flow[x-1:x+2, y-1:y+2]
      a_state = area[x-1:x+2, y-1:y+2]

      if area[x,y] < 1E2:
        continue

      E = []
      for i in range(8):
        E = np.append(E, energy(f_state, a_state, i))

      # 3. Sampling Procedure
      # -> Pick the Lowest Energy State
      # -> Update the Local State

      t = np.argmin(E)
      flow[x, y] = dirmap[t]
    
#    plt.imshow(flow != flow_next)
#    plt.show()
   # flow = flow_next

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
  plt.grid('on', zorder=0)

  plt.imshow(fdir, zorder=2, cmap='viridis')
  boundaries = ([0] + sorted(list(dirmap)))
  plt.colorbar(boundaries= boundaries,
              values=sorted(dirmap))
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.title('Flow direction grid', size=14)
  plt.tight_layout()
  plt.show()

def plot_acc(model):

  grid, acc = model
#  acc = np.transpose(acc)

  fig, ax = plt.subplots(figsize=(8,6))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)

  im = ax.imshow(acc, zorder=2, cmap='cubehelix',
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
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-72.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40718_DGM_tif_Traunkirchen/G-T4831-79.tif"
  #input = "/home/nickmcdonald/Datasets/UpperAustriaDGM/40701_DGM_tif_Altmuenster/G-T4831-52.tif"
  #input = "out_altmuenster.tiff"
  input = "out.tiff"
  main(input)