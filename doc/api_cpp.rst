C++ API
=======

Generated from the ``//!`` doc-comments in ``source/soillib`` via Doxygen and
Breathe, one section per header, grouped the way the source tree already is
rather than by C++ namespace -- everything lives in ``soil``, so the
namespace carves nothing up, while the directory layout is the real
organising structure. Internal helpers (``detail`` namespaces) and the
vendored third-party headers under ``external/`` are excluded throughout.

I/O
---

Image and mesh import/export, including floating-point TIFF and native
GeoTIFF with GDAL metadata extensions.

.. doxygenfile:: tiff.hpp

.. doxygenfile:: geotiff.hpp

.. doxygenfile:: mesh.hpp

Model
-----

The simulation models themselves, one subdirectory per family.

Filter
~~~~~~

.. doxygenfile:: filter.hpp

Gradient
~~~~~~~~

.. doxygenfile:: grad.hpp

Graph
~~~~~

Flow-graph construction and accumulation over it.

.. doxygenfile:: graph.hpp

Path
~~~~

Particle-based transport: erosion models, path integration and sampling.

.. doxygenfile:: erosion.hpp

.. doxygenfile:: path.hpp

.. doxygenfile:: sample.hpp

Operations
----------

Standalone kernelized operations over tensors.

.. doxygenfile:: noise.hpp

.. doxygenfile:: normal.hpp

Utility
-------

.. doxygenfile:: timer.hpp

.. doxygenfile:: yield.hpp
