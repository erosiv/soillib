"""
Flow-routing convenience compositions.

soillib exposes flow routing as two separate kernel stages: building a
receiver graph over a height field (direction / steepest / random_weighted),
then accumulating a source field over that graph (accumulate /
accumulate_decay). The functions here just compose those two stages for the
common "give me the discharge" case, so example code doesn't have to spell
out the graph construction every time.
"""

import silt
import soillib as soil

__all__ = ["discharge", "multiflow"]


def discharge(height, edge=None, rain=None, weighted=False, decay=None,
              seed=0, offset=0, temperature=1.0):
  """
  Flow accumulation over a height field in one call: builds the receiver
  graph (steepest-descent by default, or stochastic multi-flow when
  weighted=True) and accumulates `rain` over it.

  height:      2D height tensor (GPU-resident).
  edge:        soil.d4 or soil.d8 connectivity (default: soil.d8).
  rain:        source field to accumulate (default: uniform rainfall = 1).
  weighted:    use soil.random_weighted (single stochastic draw) instead of
               soil.steepest (deterministic steepest-descent).
  decay:       optional per-cell decay field, routed through
               soil.accumulate_decay instead of soil.accumulate.
  seed, offset, temperature: forwarded to soil.random_weighted.
  """
  if edge is None:
    edge = soil.d8

  if rain is None:
    rain = silt.ones(height.shape, host=height.host)

  if weighted:
    graph = soil.random_weighted(height, edge, seed, offset, temperature)
  else:
    graph = soil.steepest(height, edge)

  if decay is not None:
    return soil.accumulate_decay(graph, rain, decay, edge)
  return soil.accumulate(graph, rain, edge)


def multiflow(height, edge=None, rain=None, samples=512, temperature=1.0,
              seed=0):
  """
  Monte-Carlo multi-flow accumulation: averages `samples` independent
  stochastic single-flow accumulations (soil.random_weighted +
  soil.accumulate, each a different random receiver-graph draw), which
  spreads discharge across multiple downhill directions instead of
  concentrating it along one steepest-descent path.
  """
  if edge is None:
    edge = soil.d8

  if rain is None:
    rain = silt.ones(height.shape, host=height.host)

  total = silt.zeros(height.shape, host=height.host)
  for k in range(samples):
    graph = soil.random_weighted(height, edge, seed, k, temperature)
    silt.add_(total, soil.accumulate(graph, rain, edge))

  silt.multiply_(total, 1.0 / samples)
  return total
