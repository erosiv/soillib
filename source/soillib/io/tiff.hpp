#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

#include <iostream>
#include <stdexcept>
#include <stdfloat>
#include <tiffio.h>

namespace soil {
namespace io {

//! tiff is a generic .tiff file interface for reading
//! and writing generic image data to and from disk.
//!
//! tiff supports multiple different bit-depths, as
//! well as the reading of tiled .tiff images.
//!
struct tiff {

  tiff() {}
  tiff(const char *filename) { read(filename); };

  tiff(const silt::tensor &_tensor): _tensor{_tensor} {

    auto type = _tensor.type();

    this->_shape = _tensor.shape();
    this->_width = _shape[0];
    this->_height = _shape[1];

    if (type == silt::FLOAT32) {
      this->_bits = 32;
    } else if (type == silt::FLOAT64) {
      this->_bits = 64;
    }
  }

  bool peek(const char *filename);  //!< Load TIFF Metadata
  bool read(const char *filename);  //!< Read TIFF Raw Data
  bool write(const char *filename); //!< Write TIFF Raw Data

  uint32_t bits() const { return this->_bits; }
  uint32_t width() const { return this->_width; }
  uint32_t height() const { return this->_height; }

  silt::tensor tensor() const { return this->_tensor; }
  silt::shape shape() const { return this->_shape; }

protected:
  //! Flat tensor index for the file pixel at (col, row).
  //!
  //! A TIFF scanline runs west to east and row 0 is the north edge
  //! (ORIENTATION_TOPLEFT). A tensor is (W, H) with dim0 = x (east), dim1 = y
  //! (north) and y contiguous. Converting between them is y = H - row - 1,
  //! x = col.
  //!
  //! This is the only expression of that conversion: the scanline helpers and
  //! the tiled reader all call it, so read and write cannot diverge.
  inline size_t tensor_index(const size_t col, const size_t row) const {
    return col * this->_height + (this->_height - row - 1);
  }

  void scatter_scanline(const void *src, const size_t row); //!< file row -> tensor
  void gather_scanline(void *dst, const size_t row);        //!< tensor -> file row

  //! Bytes per stored element in the tensor, which is what a staging buffer is
  //! sized from.
  inline size_t elem_size() const { return (this->_bits == 64) ? 8 : 4; }

  bool meta_loaded = false; //!< Flag: Is Meta-Data Loaded
  bool tiled_image = false; //!< Flag: Is Image Tiled

  std::string filename; //!< Original Filename

  uint32_t _width = 0;  //!< Image Width
  uint32_t _height = 0; //!< Image Height
  uint32_t _bits = 0;   //!< Pixel Bit-Depth

  uint32_t _twidth = 0;  //!< Tile Width
  uint32_t _theight = 0; //!< Tile Height

  silt::shape _shape;   //!< Underlying Data Shape
  silt::tensor _tensor; //!< Underlying Data Tensor
};

//! Scatter one file scanline into the tensor, reversing y.
void tiff::scatter_scanline(const void *src, const size_t row) {

  void *data = this->_tensor.data();

  for (size_t x = 0; x < this->width(); ++x) {
    const size_t i = this->tensor_index(x, row);
    if (this->bits() == 32) {
      ((float *)data)[i] = ((const float *)src)[x];
    }
    if (this->bits() == 64) {
      ((double *)data)[i] = ((const double *)src)[x];
    }
  }
}

//! Gather one file scanline out of the tensor, reversing y.
void tiff::gather_scanline(void *dst, const size_t row) {

  const void *data = this->_tensor.data();

  for (size_t x = 0; x < this->width(); ++x) {
    const size_t i = this->tensor_index(x, row);
    if (this->bits() == 32) {
      ((float *)dst)[x] = ((const float *)data)[i];
    }
    if (this->bits() == 64) {
      ((double *)dst)[x] = ((const double *)data)[i];
    }
  }
}

//! Load TIFF Metadata
bool tiff::peek(const char *filename) {

  TIFF *tif = TIFFOpen(filename, "r");
  if (tif == NULL) {
    throw silt::error::missing_file(filename);
    return false;
  }

  if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &this->_width))
    return false;

  if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &this->_height))
    return false;

  if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &this->_bits))
    return false;

  if (TIFFGetField(tif, TIFFTAG_TILEWIDTH, &this->_twidth))
    this->tiled_image = true;

  if (TIFFGetField(tif, TIFFTAG_TILELENGTH, &this->_theight))
    this->tiled_image = true;

  TIFFClose(tif);

  this->filename = filename;
  this->meta_loaded = true;
  return true;
}

//! Read TIFF Raw Data
bool tiff::read(const char *filename) {

  if (!meta_loaded) {
    this->peek(filename);
  }

  // Only 32- and 64-bit float samples are read. Widening a 16-bit file would
  // need silt to carry the source depth to write it back unchanged, so it is
  // rejected rather than silently returning the wrong type or empty data.
  if (this->bits() != 32 && this->bits() != 64) {
    throw std::runtime_error("TIFF bit depth not supported: " +
                             std::to_string(this->bits()) +
                             " (only 32 and 64 bit float samples are read)");
  }

  // Shape is (W, H): dim0 = x (east), dim1 = y (north), y contiguous. The file
  // is row-major with row 0 north, so the scanline helpers reverse y on the way
  // in and out. That conversion happens here and nowhere else.

  this->_shape = silt::shape(this->width(), this->height());

  if (this->bits() == 32) {
    this->_tensor = silt::tensor(silt::FLOAT32, this->_shape);
  }
  if (this->bits() == 64) {
    this->_tensor = silt::tensor(silt::FLOAT64, this->_shape);
  }

  TIFF *tif = TIFFOpen(filename, "r");
  if (tif == NULL) {
    throw silt::error::missing_file(filename);
    return false;
  }

  // Load Tiled / Non-Tiled Images
  if (!this->tiled_image) {

    // Holds one FILE scanline, so this one really is bits()/8.
    uint8_t *nbuf = new uint8_t[this->width() * (this->bits() / 8)];

    for (size_t row = 0; row < this->height(); row++) {
      TIFFReadScanline(tif, nbuf, row);
      this->scatter_scanline(nbuf, row);
    }

    delete[] nbuf;
  }

  else {

    const size_t tsize = this->_twidth * this->_theight;
    const size_t nwidth = (this->width() + this->_twidth - 1) / this->_twidth;
    const size_t nheight = (this->height() + this->_theight - 1) / this->_theight;

    auto data = this->_tensor.data();
    uint8_t *buf = (uint8_t *)data;

    uint8_t *nbuf = new uint8_t[tsize * (this->bits() / 8)];

    for (size_t nx = 0; nx < nwidth; ++nx)
      for (size_t ny = 0; ny < nheight; ++ny) {

        const glm::ivec2 npos(nx, ny);
        const glm::ivec2 norg = npos * glm::ivec2(this->_twidth, this->_theight);

        if (!TIFFReadTile(tif, nbuf, norg.x, norg.y, 0, 0)) {
          continue;
        }

        for (size_t ix = 0; ix < this->_twidth; ix++)
          for (size_t iy = 0; iy < this->_theight; iy++) {

            glm::ivec2 tpos = glm::ivec2(ix, iy);
            glm::ivec2 fpos = norg + tpos;

            if (fpos.x >= this->width())
              continue;
            if (fpos.y >= this->height())
              continue;

            // Reverses y, as the scanline helpers do.
            const size_t i = this->tensor_index(fpos.x, fpos.y);

            if (this->bits() == 32) {
              ((float *)buf)[i] = ((float *)nbuf)[iy * this->_twidth + ix];
            }
            if (this->bits() == 64) {
              ((double *)buf)[i] = ((double *)nbuf)[iy * this->_twidth + ix];
            }
          }
      }

    delete[] nbuf;
  }

  TIFFClose(tif);
  return true;
}

bool tiff::write(const char *filename) {

  TIFF *out = TIFFOpen(filename, "w");

  TIFFSetField(out, TIFFTAG_IMAGEWIDTH, this->width());
  TIFFSetField(out, TIFFTAG_IMAGELENGTH, this->height());
  TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, this->bits());
  TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, this->width()));

  // Sized from the stored element, not bits()/8: see elem_size.
  uint8_t *nbuf = new uint8_t[this->width() * this->elem_size()];

  for (uint32_t row = 0; row < this->height(); row++) {
    this->gather_scanline(nbuf, row);
    if (TIFFWriteScanline(out, nbuf, row, 0) < 0)
      break;
  }

  delete[] nbuf;

  TIFFClose(out);
  return true;
}

}; // end of namespace io
}; // end of namespace soil
