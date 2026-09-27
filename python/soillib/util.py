"""
Generic tensor / GeoTIFF display and I/O convenience helpers.
"""

import os
from zipfile import ZipFile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import silt
import soillib as soil

__all__ = [
  "iter_tiff",
  "relief_shade",
  "plot_area",
  "plot_images",
  "show_height",
  "show_normal",
  "show_relief",
  "show_discharge",
  "show_mass",
  "zip_save",
]


def iter_tiff(path, max_files=None):
  """Generator over all files in a directory, or a single file."""

  path = os.fsencode(path)
  if not os.path.exists(path):
    raise RuntimeError("path does not exist")

  if os.path.isfile(path):
    file = os.path.basename(path)
    yield file.decode('utf-8'), path.decode('utf-8')

  elif os.path.isdir(path):
    for k, file in enumerate(os.listdir(path)):
      if max_files is not None and k > max_files:
        break
      yield file.decode('utf-8'), os.path.join(path, file).decode('utf-8')

  else:
    raise RuntimeError("path must be file or directory")


def relief_shade(h, n):
  """Diffuse hillshade from a height array `h` and a normal field `n`."""

  h_min = np.nanmin(h)
  h_max = np.nanmax(h)
  h = (h - h_min) / (h_max - h_min)

  light = np.array([-1, 2, 1])
  light = light / np.linalg.norm(light)

  diffuse = np.sum(light * n, axis=-1)

  flattone = np.full(h.shape, 0.75)
  weight = 1.0

  return weight * diffuse + (1.0 - weight) * flattone


def _to_cpu_numpy(tensor):
  """A CPU numpy view of `tensor`, without mutating the caller's tensor."""
  if tensor.host == silt.cpu:
    return tensor.numpy()
  return tensor.copy_to(silt.cpu).numpy()


def plot_area(area):
  """Log-scale upstream-cell-count plot for a flow-accumulation array."""

  fig, ax = plt.subplots(figsize=(8, 6))
  fig.patch.set_alpha(0)
  plt.grid('on', zorder=0)
  im = ax.imshow(area, zorder=2,
                  cmap='CMRmap',
                  norm=colors.LogNorm(1, area.max()),
                  interpolation='bilinear')
  plt.colorbar(im, ax=ax, label='Upstream Cells')
  plt.tight_layout()
  plt.show()


def plot_images(images):
  """A row of images side by side, e.g. for comparing pipeline stages."""

  K = len(images)
  fig, ax = plt.subplots(1, K, figsize=(4 * K, 4))
  fig.patch.set_alpha(0)
  if K == 1:
    ax = [ax]
  for k, img in enumerate(images):
    ax[k].imshow(img, zorder=2, cmap='CMRmap', interpolation='bilinear')
  plt.tight_layout()
  plt.show()


def show_height(tensor):
  plt.imshow(_to_cpu_numpy(tensor))
  plt.show()


def show_normal(tensor, scale):
  """`tensor` should be GPU-resident (soil.normal runs on the GPU)."""
  normal = _to_cpu_numpy(soil.normal(tensor, scale))
  plt.imshow(0.5 + 0.5 * normal)
  plt.show()


def show_relief(tensor, scale):
  """Hillshade a height tensor. `tensor` should be GPU-resident."""
  normal = _to_cpu_numpy(soil.normal(tensor, scale))
  height = _to_cpu_numpy(tensor)
  relief = relief_shade(height, normal)
  plt.imshow(relief, cmap='gray')
  plt.show()


def show_discharge(tensor):
  data = 1.0 + _to_cpu_numpy(tensor)
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.imshow(data, zorder=2,
            cmap='CMRmap',
            norm=colors.LogNorm(1, data.max()),
            interpolation='none')
  plt.show()


def show_mass(tensor):
  data = 1.0 + _to_cpu_numpy(tensor)
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.imshow(data, zorder=2,
            cmap='CMRmap',
            norm=colors.LogNorm(1, data.max()),
            interpolation='none')
  plt.show()


def zip_save(output, fields, scale):
  """Save a dict of {name: tensor} as GeoTIFFs, using `scale` as the pixel
  scale, bundled into a single zip archive."""
  with ZipFile(output, 'w') as archive:
    for name, field in fields.items():
      filename = f"{name}.tiff"
      tiff_out = soil.geotiff(field.copy_to(silt.cpu))
      tiff_out.meta.scale = scale
      tiff_out.write(filename)
      archive.write(filename)
      os.remove(filename)
