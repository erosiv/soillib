Conventions
===========

Layout and orientation of 2D field data.

Tensor layout
-------------

A 2D field is a ``silt`` tensor of shape ``(W, H)`` or ``(W, H, C)``:

.. code-block:: text

   dim0 = X   east,  0 .. W-1
   dim1 = Y   north, 0 .. H-1
   dim2 = C   channel

The flat index is ``silt::shape::flatten``, which makes the last dimension
contiguous:

.. code-block:: text

   index = (x * H + y) * C + c

Y is therefore the fast axis: a field is ``W`` columns, each holding ``H``
contiguous values running south to north. ``flatten`` and ``unflatten`` are the
only supported way to index a field.

Frames
------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Frame
     - Axes
     - Origin
   * - Tensor
     - ``+x`` east, ``+y`` north
     - cell ``(0, 0)``, south-west
   * - World
     - ``+x`` east, ``+y`` north
     - set by the field's ``origin``
   * - Image file
     - ``+col`` east, ``+row`` south
     - row 0, top-left

Tensor and world space agree on direction, so a field placed in the world is

.. code-block:: text

   world = origin + cell * (x, y)

where ``origin`` is the south-west corner and ``cell`` the per-axis cell size.
For a GeoTIFF these are ``geotiff::min()`` and ``geotiff::scale()``.

Image files are the one frame that differs, which is what the boundaries below
convert.

Implications
------------

- ``cell.x`` pairs with dim0 and ``cell.y`` with dim1. A parameter indexed
  ``[0]`` addresses dim0, which is X.
- A gradient's ``.x`` is the derivative along dim0, i.e. eastward. A normal from
  ``soil::normal`` is ``(-dh/dx, -dh/dy, 1)`` with ``z`` up.
- ``silt`` dimension order matches numpy C-order: axis 0 is dim0 and is the
  slowest. A ``(W, H)`` numpy array round-trips without a transpose. A field
  consequently prints transposed relative to how an image viewer displays it;
  use ``soillib.as_image`` for display rather than transposing the data.
- On a square field this layout and a row-major image layout produce identical
  indices, so ``W != H`` is required to distinguish them.

Boundaries
----------

A transform between frames occurs only in the following places. Everywhere else
indexes through ``flatten``.

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - Boundary
     - Location
     - Transform
   * - TIFF / GeoTIFF
     - ``io/tiff.hpp``: ``tensor_index``
     - ``tensor(x, y, c) = file(row = H-1-y, col = x, sample = c)``
   * - PNG
     - ``soillib.load_png``
     - the same, via ``from_image``
   * - Viewers, image writers
     - ``soillib.as_image`` / ``soillib.from_image``
     - tensor order to ``(rows, cols)`` image order

``io/tiff.hpp`` expresses the mapping once, in ``tensor_index``; the scanline
helpers and the tiled reader both call it, so read and write cannot diverge.

``as_image`` and ``from_image`` are an exact inverse pair, and are the only
conversion that should appear before a ``matplotlib`` ``imshow`` or a
``PIL.Image.fromarray``.

Channels
--------

A tensor channel is one TIFF **sample per pixel**, not one page. A ``(W, H, C)``
tensor writes as a single image with ``SAMPLESPERPIXEL = C`` and
``PLANARCONFIG_CONTIG``, which interleaves the samples of a pixel in the same
order the tensor's flat index does. The file conversion is therefore the 2D one
with a sample stride, and no repacking occurs.

``SAMPLESPERPIXEL`` requires every channel to share ``W``, ``H``, bit depth and
sample format. Channels past the first are tagged ``EXTRASAMPLE_UNSPECIFIED``,
not alpha, so a reader does not premultiply them.

Reading accepts ``PLANARCONFIG_CONTIG`` and ``PLANARCONFIG_SEPARATE``, tiled and
stripped. A file with ``SAMPLESPERPIXEL = 1`` loads as a 2D ``(W, H)`` tensor,
since ``silt::shape`` drops trailing singleton dimensions.

Georeferencing
--------------

A GeoTIFF ``ModelTiepointTag`` gives the **north-west** corner, and
``ModelPixelScale`` is positive with the row index increasing southward, so
world ``y`` decreases down the raster.

``geotiff::min()`` is therefore the south-west corner — tensor cell ``(0, 0)`` —
and ``geotiff::max()`` the north-east. The tie point's ``y`` is the maximum.
``meta_t::set_min`` converts a south-west corner back to the stored tie point
and needs the cell size, so assign ``scale`` before ``min`` when writing.
