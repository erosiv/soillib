"""
Flow-routing convenience compositions.

soillib exposes flow routing as two separate kernel stages: building a
receiver graph over a height field (steepest / random_weighted), then
accumulating a source field over that graph (accumulate /
accumulate_decay). The functions here just compose those two stages for the
common "give me the discharge" case, so example code doesn't have to spell
out the graph construction every time.

soil.random_weighted draws each cell's receiver with probability
proportional to its downhill slope, so a single draw is still a
single-receiver graph, but the expectation over draws is the slope-weighted
MFD partition -- see multiflow below.
"""

import silt
import soillib as soil

__all__ = ["discharge", "multiflow"]


def discharge(height, edge=None, rain=None, decay=None, rng=None):
  """
  Flow accumulation over a height field in one call: builds the receiver
  graph and accumulates `rain` over it.

  height: 2D height tensor (GPU-resident).
  edge:   soil.d4 or soil.d8 connectivity (default: soil.d8).
  rain:   source field to accumulate (default: uniform rainfall = 1).
  decay:  optional per-cell decay field, routed through
          soil.accumulate_decay instead of soil.accumulate.
  rng:    optional RNG-state tensor (silt.rand). Passing one switches the
          graph from deterministic steepest-descent to a single
          slope-weighted stochastic draw, and advances the state in place.
  """
  if edge is None:
    edge = soil.d8

  if rain is None:
    rain = silt.ones(height.shape, host=height.host)

  if rng is not None:
    graph = soil.random_weighted(height, edge, rng)
  else:
    graph = soil.steepest(height, edge)

  if decay is not None:
    return soil.accumulate_decay(graph, rain, decay, edge)
  return soil.accumulate(graph, rain, edge)


def multiflow(height, edge=None, rain=None, samples=512, rng=None, seed=0):
  """
  Monte-Carlo multi-flow accumulation: averages `samples` independent
  slope-weighted stochastic single-flow accumulations.

  Since soil.random_weighted samples receivers proportionally to slope, the
  average converges to the slope-weighted MFD discharge field, while every
  individual draw stays a single-receiver graph the O(N) accumulation can
  consume. This spreads discharge across all downhill directions and
  removes the directional artifacts a deterministic D4/D8 graph imprints on
  divergent terrain.

  rng:  optional RNG-state tensor to draw from (default: allocate and seed
        one from `seed`). The state advances in place, so each sample is an
        independent draw -- pass your own to control or continue the
        sequence across calls.
  """
  if edge is None:
    edge = soil.d8

  if rain is None:
    rain = silt.ones(height.shape, host=height.host)

  if rng is None:
    rng = silt.rand(height.shape, seed)

  total = silt.zeros(height.shape, host=height.host)
  for _ in range(samples):
    graph = soil.random_weighted(height, edge, rng)
    silt.add_(total, soil.accumulate(graph, rain, edge))

  silt.multiply_(total, 1.0 / samples)
  return total
