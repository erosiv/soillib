"""
GPU stochastic erosion model -- a thin, reusable convenience layer over
soillib's individual transport / mass-transfer kernels (transport_fluvial,
transport_debris, mass_transfer, layer_merge). Those kernels are the
"official" implementation referenced in soillib's README; this module is
just the tensor bookkeeping and step choreography around them, modeled on
erosiv's own reference erosion node, trimmed to what an example script
needs (no albedo/visual channels -- those are a rendering concern, not part
of the erosion physics).

`layers` is a single 2-component tensor per soillib's own convention
(soil.layer_merge takes one `layers` argument, not two): component 0 is
taken here as bedrock elevation and component 1 as sediment thickness, with
soil.layer_merge collapsing the two into a single height field.
"""

import numpy as np
import silt
import soillib as soil

__all__ = ["ErosionModel", "plot"]


class ErosionModel:
  """
  A stochastic geotransport erosion model over a single height field.

  Tracks a two-layer [bedrock, sediment] height stack, fluvial
  discharge/mass/momentum and debris-flow mass/momentum, and steps them
  forward with soillib's particle-based transport kernels.
  """

  def __init__(self, shape, scale, param=None, seed=0, samples=8192 * 16):
    """
    shape:   silt.shape (2D) -- simulation resolution.
    scale:   [dx, dy, dz] world-space pixel scale, e.g.
             [20.0/256, 20.0/256, 4.0] (km/px in x and y, height unit in z).
    param:   soil.param_t; None falls back to soillib's own defaults.
    samples: particle sample count for the transport kernels' RNG pool.
    """
    self.shape = shape
    self.scale = list(scale)
    self.param = param if param is not None else soil.param_t()
    self.age = 0

    def z1():
      return silt.zeros(shape, host=silt.gpu)

    def z2():
      return silt.zeros(silt.shape(shape[0], shape[1], 2), host=silt.gpu)

    def z3():
      return silt.zeros(silt.shape(shape[0], shape[1], 3), host=silt.gpu)

    self.layers = z2()
    self.height = z1()

    self.rainfall = silt.ones(shape, host=silt.gpu)
    self.uplift = z1()

    self.discharge = z1()
    self.discharge_track = z1()
    self.mass = z1()
    self.mass_track = z1()
    self.momentum = z2()
    self.momentum_track = z2()

    self.debris = z1()
    self.debris_track = z1()
    self.debris_momentum = z2()
    self.debris_momentum_track = z2()

    self._delta = z2()

    # transport_fluvial/transport_debris require albedo channels
    # structurally, even though this convenience layer never reads them
    # back out -- plain scratch buffers here, not exposed for tuning.
    self._albedo_bedrock = z3()
    self._albedo_surface = z3()
    self._albedo_transport_fluvial = z3()
    self._albedo_transport_debris = z3()

    self.rng = silt.tensor(silt.rng, silt.shape(samples), silt.gpu)
    silt.seed(self.rng, seed, 0)

  def set_height(self, height):
    """Seed the bedrock layer from a height field (sediment starts at 0)."""
    if hasattr(height, "numpy"):
      cpu = height if height.host == silt.cpu else height.copy_to(silt.cpu)
      height = cpu.numpy()

    layers_np = np.zeros((*height.shape, 2), dtype=np.float32)
    layers_np[..., 0] = height
    self.layers = silt.tensor.from_numpy(layers_np).to_gpu()
    soil.layer_merge(self.height, self.layers)

  def set_uplift(self, uplift):
    """Set the uplift-rate field (numpy array or tensor, GPU-resident)."""
    if hasattr(uplift, "to_gpu"):
      self.uplift = uplift if uplift.host == silt.gpu else uplift.copy_to(silt.gpu)
    else:
      self.uplift = silt.tensor.from_numpy(np.asarray(uplift, dtype=np.float32)).to_gpu()

  def step(self, substeps=1):
    """Advance the simulation by `substeps` transport + mass-transfer passes."""
    for _ in range(substeps):

      silt.set_(self.discharge_track, 0.0)
      silt.set_(self.mass_track, 0.0)
      silt.set_(self.momentum_track, 0.0)
      silt.set_(self.debris_track, 0.0)
      silt.set_(self.debris_momentum_track, 0.0)

      if self.param.suspensionRateFluvial > 0:
        soil.transport_fluvial(
          self.layers, self.rainfall,
          self.discharge, self.discharge_track,
          self.mass, self.mass_track,
          self.momentum, self.momentum_track,
          self._albedo_bedrock, self._albedo_transport_fluvial, self._albedo_surface,
          self.rng, self.scale, self.param,
        )

      if self.param.landslideRateDebris > 0:
        soil.transport_debris(
          self.layers,
          self.debris_momentum, self.debris_momentum_track,
          self.debris, self.debris_track,
          self._albedo_bedrock, self._albedo_transport_debris, self._albedo_surface,
          self.rng, self.scale, self.param,
        )

      silt.set_(self._delta, 0.0)
      soil.mass_transfer(
        self._delta, self.layers, self.uplift,
        self.discharge, self.mass, self.momentum,
        self.debris, self.debris_momentum,
        self._albedo_bedrock, self._albedo_transport_fluvial, self._albedo_transport_debris, self._albedo_surface,
        self.scale, self.param,
      )
      silt.add_(self.layers, self._delta)

      self.age += 1

    soil.layer_merge(self.height, self.layers)
    return self.height

  def resized(self, new_shape, scale=None):
    """
    A new ErosionModel at `new_shape`, carrying over this model's state
    (height/sediment layers, discharge, mass, momentum, debris, uplift,
    rainfall) via soil.resize. Used to step a simulation up through a
    sequence of resolutions without restarting it from scratch.
    """
    new = ErosionModel(new_shape, scale if scale is not None else self.scale, self.param)

    new.layers = soil.resize(self.layers, new_shape)
    new.rainfall = soil.resize(self.rainfall, new_shape)
    new.uplift = soil.resize(self.uplift, new_shape)
    new.discharge = soil.resize(self.discharge, new_shape)
    new.mass = soil.resize(self.mass, new_shape)
    new.momentum = soil.resize(self.momentum, new_shape)
    new.debris = soil.resize(self.debris, new_shape)
    new.debris_momentum = soil.resize(self.debris_momentum, new_shape)

    soil.layer_merge(new.height, new.layers)
    new.age = self.age
    return new


def plot(model):
  """
  Bedrock-vs-sediment hillshade of an ErosionModel: cyan where sediment
  has accumulated, red where bedrock is still exposed.
  """
  import matplotlib.pyplot as plt
  from . import util

  layers = model.layers.copy_to(silt.cpu).numpy()
  height = layers[..., 0] + layers[..., 1]
  sediment = layers[..., 1]

  normal = soil.normal(model.height, model.scale).copy_to(silt.cpu).numpy()
  relief = util.relief_shade(height, normal)
  relief = 0.5 + 0.5 * relief

  shaded = np.repeat(relief[..., None], 3, axis=-1)
  shaded[sediment >= 1e-4] *= [0.0, 1.0, 1.0]
  shaded[sediment < 1e-4] *= [1.0, 0.0, 0.0]

  plt.imshow(shaded, interpolation='bilinear')
  plt.show()
