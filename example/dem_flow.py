#!/usr/bin/env python

"""
Flow-Routing Example

Showcases soillib's flow-routing kernels on a DEM: building a receiver
graph over the height field, then accumulating a source field over it
(soil.flow.discharge / soil.flow.multiflow -- see soil/flow.py).

Compares three routing strategies side by side:
  - Single-flow: deterministic steepest-descent (soil.steepest)
  - Decayed:     single-flow with a decay field (soil.accumulate_decay),
                 e.g. for evaporation-limited discharge
  - Multi-flow:  Monte-Carlo average of stochastic single-flow draws
                 (soil.random_weighted), spreading discharge across
                 multiple downhill directions instead of one path
"""

import soillib as soil
import silt

import matplotlib.pyplot as plt
from matplotlib import colors


def main(data):

  tiff = soil.geotiff(data)
  height = tiff.tensor.to_gpu()

  timer = soil.timer(soil.ms)

  with timer:
    single = soil.flow.discharge(height)
  print(f"Single-Flow:  {timer.count} ms")

  with timer:
    decay = silt.full(height.shape, 0.98, host=silt.gpu)
    decayed = soil.flow.discharge(height, decay=decay)
  print(f"Decayed Flow: {timer.count} ms")

  with timer:
    multi = soil.flow.multiflow(height, samples=256)
  print(f"Multi-Flow:   {timer.count} ms")

  fig, ax = plt.subplots(1, 4, figsize=(20, 5))
  fig.suptitle("Flow Routing: Single-Flow vs. Decayed vs. Multi-Flow")

  ax[0].imshow(height.copy_to(silt.cpu).numpy(), cmap='terrain')
  ax[0].set_title("Height")

  panels = [
    ("Single-Flow (D8)", single),
    ("Single-Flow w. Decay", decayed),
    ("Multi-Flow (256 samples)", multi),
  ]

  for a, (title, field) in zip(ax[1:], panels):
    field_np = field.copy_to(silt.cpu).numpy()
    a.imshow(field_np, cmap='CMRmap', norm=colors.LogNorm(1, field_np.max()), interpolation='none')
    a.set_title(title)

  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  main("data/dem_1024.tiff")
