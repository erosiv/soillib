"""
GPU stochastic erosion model -- a thin, reusable convenience layer over
soillib's individual transport / mass-transfer kernels (transport_fluvial,
transport_debris, mass_transfer, layer_merge). Those kernels are the
"official" implementation referenced in soillib's README; this module is
just the tensor bookkeeping and step choreography around them.

`layers` is a single 2-component tensor per soillib's own convention
(soil.layer_merge takes one `layers` argument, not two): component 0 is
taken here as bedrock elevation and component 1 as sediment thickness, with
soil.layer_merge collapsing the two into a single height field.
"""

import os

import numpy as np
import silt
import soillib as soil

__all__ = ["ErosionModel", "plot", "export"]


class ErosionModel:
  """
  A stochastic geotransport erosion model over a single height field.

  Tracks a two-layer [bedrock, sediment] height stack, fluvial
  discharge/mass/momentum and debris-flow mass/momentum, surface/bedrock
  albedo, and steps them forward with soillib's particle-based transport
  kernels.
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

    self.albedo_bedrock = z3()
    self.albedo_surface = z3()
    # Internal transport-kernel scratch: per-step albedo accumulation, not
    # meaningful state on its own outside a step() call.
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

  def set_rainfall(self, rainfall):
    """Set the rainfall-rate field (numpy array or tensor, GPU-resident)."""
    if hasattr(rainfall, "to_gpu"):
      self.rainfall = rainfall if rainfall.host == silt.gpu else rainfall.copy_to(silt.gpu)
    else:
      self.rainfall = silt.tensor.from_numpy(np.asarray(rainfall, dtype=np.float32)).to_gpu()

  def set_albedo_bedrock(self, albedo):
    """
    Set the bedrock albedo field (3-channel numpy array or tensor,
    GPU-resident). Also resets the surface albedo to match, mirroring
    freshly-allocated model state before any sediment has been deposited.
    """
    if hasattr(albedo, "to_gpu"):
      albedo = albedo if albedo.host == silt.gpu else albedo.copy_to(silt.gpu)
    else:
      albedo = silt.tensor.from_numpy(np.asarray(albedo, dtype=np.float32)).to_gpu()
    self.albedo_bedrock = albedo
    silt.set_(self.albedo_surface, albedo)

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
          self.albedo_bedrock, self._albedo_transport_fluvial, self.albedo_surface,
          self.rng, self.scale, self.param,
        )

      if self.param.landslideRateDebris > 0:
        soil.transport_debris(
          self.layers,
          self.debris_momentum, self.debris_momentum_track,
          self.debris, self.debris_track,
          self.albedo_bedrock, self._albedo_transport_debris, self.albedo_surface,
          self.rng, self.scale, self.param,
        )

      silt.set_(self._delta, 0.0)
      soil.mass_transfer(
        self._delta, self.layers, self.uplift,
        self.discharge, self.mass, self.momentum,
        self.debris, self.debris_momentum,
        self.albedo_bedrock, self._albedo_transport_fluvial, self._albedo_transport_debris, self.albedo_surface,
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
    new.albedo_bedrock = soil.resize(self.albedo_bedrock, new_shape)
    new.albedo_surface = soil.resize(self.albedo_surface, new_shape)

    soil.layer_merge(new.height, new.layers)
    new.age = self.age
    return new


def plot(model):
  """
  A 2x2 summary of an ErosionModel: surface normal coloration, surface
  albedo, relief-shaded surface albedo, and the discharge map.
  """
  import matplotlib.pyplot as plt
  from matplotlib import colors
  from . import util

  layers = model.layers.copy_to(silt.cpu).numpy()
  height = layers[..., 0] + layers[..., 1]

  normal = soil.normal(model.height, model.scale).copy_to(silt.cpu).numpy()
  normal_rgb = np.clip(0.5 + 0.5 * normal, 0.0, 1.0)

  albedo = util._to_cpu_numpy(model.albedo_surface)
  albedo = np.clip(albedo, 0.0, 1.0)

  diffuse = np.clip(0.5 + 0.5 * util.relief_shade(height, normal), 0.0, 1.0)
  shaded_albedo = np.clip(diffuse[..., None] * albedo, 0.0, 1.0)

  discharge = util._to_cpu_numpy(model.discharge)

  fig, ax = plt.subplots(2, 2, figsize=(12, 12))

  # as_image transposes and reverses y: tensor order -> image order.
  ax[0, 0].imshow(soil.util.as_image(normal_rgb))
  ax[0, 0].set_title("Surface Normal")

  ax[0, 1].imshow(soil.util.as_image(albedo))
  ax[0, 1].set_title("Surface Albedo")

  ax[1, 0].imshow(soil.util.as_image(shaded_albedo))
  ax[1, 0].set_title("Relief-Shaded Albedo")

  ax[1, 1].imshow(soil.util.as_image(1.0 + discharge), cmap='CMRmap',
                   norm=colors.LogNorm(1, 1.0 + np.nanmax(discharge)))
  ax[1, 1].set_title("Discharge")

  for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])

  plt.tight_layout()
  plt.show()


def export(model, out_dir="out"):
  """
  Export every tensor tracked by an ErosionModel as individual image
  files under out_dir: a float32 GeoTIFF per single-channel field, an
  x/y GeoTIFF pair per 2-component vector field, and an 8-bit PNG per
  3-channel color field -- see soil.util.save_tensor.
  """
  from . import util

  os.makedirs(out_dir, exist_ok=True)

  layers = model.layers.copy_to(silt.cpu).numpy()
  bedrock = silt.tensor.from_numpy(np.ascontiguousarray(layers[..., 0]))
  sediment = silt.tensor.from_numpy(np.ascontiguousarray(layers[..., 1]))

  normal = soil.normal(model.height, model.scale)
  silt.multiply_(normal, 0.5)
  silt.add_(normal, 0.5)

  fields = {
    "height": model.height,
    "bedrock": bedrock,
    "sediment": sediment,
    "discharge": model.discharge,
    "mass": model.mass,
    "debris": model.debris,
    "momentum": model.momentum,
    "debris_momentum": model.debris_momentum,
    "uplift": model.uplift,
    "rainfall": model.rainfall,
    "albedo_bedrock": model.albedo_bedrock,
    "albedo_surface": model.albedo_surface,
    "normal": normal,
  }

  written = []
  for name, tensor in fields.items():
    written += util.save_tensor(os.path.join(out_dir, name), tensor, scale=model.scale)
  return written
