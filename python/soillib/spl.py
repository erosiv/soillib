"""
Stream-power erosion, composed from soillib's per-cell SPL kernels
(spl_rate, spl_deposition, spl_diffusivity) and its flow-routing / blur
primitives (steepest, accumulate, slope, gaussian_blur).

Suspension - deposition - transport - blur, one step:

  1. graph, Qw  -- water discharge, via the existing flow-routing kernels.
  2. E = spl_rate(Qw, S, ...)          -- detachment (stream-power incision).
  3. Qs = accumulate(graph, E, edge)   -- sediment discharge: E is the
     source field fed through the *same* accumulation soil.accumulate uses
     for water, so it also works on an RGB source (see soil.accumulate).
  4. D = spl_deposition(Qs, Qw, k_S)   -- k_S * Qs / Qw, concentration-based.
  5. height += dt * (D - E)            -- net surface change.
  6. diffusivity = spl_diffusivity(Qw, S, ...); blur height and color by it
     -- a fast local-redistribution stand-in for the sediment transport
     FastFlow's single-receiver routing can't resolve in flat areas, and
     the mechanism by which color rides along with redistributed material.

Kept as plain functions, not a stateful model class: callers own their own
height/color/discharge tensors and drive the timestep loop themselves --
this mirrors flow.py, not erosion.py's heavier ErosionModel.
"""

import silt
import soillib as soil

__all__ = ["step"]


def step(
  height, color, graph_scale, edge,
  rainfall,
  K, m, n,
  k_S,
  D0, k_D,
  dt,
):
  """
  One suspension-deposition-transport-blur step, in place on `height` and
  `color`.

  height:      (X, Y) elevation, GPU-resident, updated in place.
  color:       (X, Y, 3) surface color, GPU-resident, updated in place
               (blurred alongside height -- see module docstring).
  graph_scale: [dx, dy, dz] world-space pixel scale, used for both the
               flow graph's slope and gaussian_blur's diffusivity.
  edge:        soil.d4 or soil.d8.
  rainfall:    (X, Y) water source field for the discharge accumulation.
  K, m, n:     stream-power incision parameters: E = K * Qw^m * S^n.
  k_S:         deposition rate constant: D = k_S * Qs / Qw.
  D0, k_D:     diffusivity base + stream-power-scaled term for the blur
               pass: sigma = D0 + k_D * Qw^m * S^n.
  dt:          forward-Euler timestep.

  Returns (Qw, Qs, E, D) for inspection/plotting.
  """
  graph = soil.steepest(height, edge)
  Qw = soil.accumulate(graph, rainfall, edge)
  S = soil.slope(height, graph, graph_scale)

  E = soil.spl_rate(Qw, S, K=K, m=m, n=n)
  Qs = soil.accumulate(graph, E, edge)
  D = soil.spl_deposition(Qs, Qw, k_S=k_S)

  net = silt.add(E, silt.multiply(D, -1.0))  # net erosion = E - D
  soil.spl_apply_(height, net, dt=dt)        # height -= dt * (E - D)

  diffusivity = soil.spl_diffusivity(Qw, S, D0=D0, k_D=k_D, m=m, n=n)
  height = soil.gaussian_blur(height, diffusivity)
  color = soil.gaussian_blur(color, diffusivity)

  return height, Qw, Qs, E, D
