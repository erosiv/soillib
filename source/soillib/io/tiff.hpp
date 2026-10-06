#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

#include <iostream>
#include <stdexcept>
#include <stdfloat>
#include <tiffio.h>
#include <vector>

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

    const int dim = this->_shape.dim();
    if (dim > 3)
      throw std::runtime_error("TIFF shape must be (W, H) or (W, H, C), got " +
                               std::to_string(dim) + " dimensions");
    this->_samples = (dim == 3) ? this->_shape[2] : 1;

    if (type == silt::FLOAT32) {
      this->_bits = 32;
    } else if (type == silt::FLOAT64) {
      this->_bits = 64;
    } else {
      throw std::runtime_error("TIFF sample type not supported (only float32 and float64 are written)");
    }
  }

  bool peek(const char *filename);  //!< Load TIFF Metadata
  bool read(const char *filename);  //!< Read TIFF Raw Data
  bool write(const char *filename); //!< Write TIFF Raw Data

  uint32_t bits() const { return this->_bits; }
  uint32_t width() const { return this->_width; }
  uint32_t height() const { return this->_height; }
  uint32_t samples() const { return this->_samples; } //!< Samples per Pixel, i.e. tensor channels

  silt::tensor tensor() const { return this->_tensor; }
  silt::shape shape() const { return this->_shape; }

protected:
  //! Flat tensor index for sample `sample` of the file pixel at (col, row).
  //!
  //! A TIFF scanline runs west to east and row 0 is the north edge
  //! (ORIENTATION_TOPLEFT). A tensor is (W, H[, C]) with dim0 = x (east),
  //! dim1 = y (north) and the channel contiguous. Converting between them is
  //! y = H - row - 1, x = col, c = sample.
  //!
  //! This is the only expression of that conversion: the scanline helpers and
  //! the tiled reader all call it, so read and write cannot diverge.
  inline size_t tensor_index(const size_t col, const size_t row, const size_t sample = 0) const {
    return (col * this->_height + (this->_height - row - 1)) * this->_samples + sample;
  }

  //! file row -> tensor. `sample` names the plane for PLANARCONFIG_SEPARATE,
  //! or is -1 for a scanline holding every sample interleaved.
  void scatter_scanline(const void *src, const size_t row, const int sample = -1);

  //! tensor -> file row, always interleaved: output is PLANARCONFIG_CONTIG.
  void gather_scanline(void *dst, const size_t row);

  //! file tile with origin `org` -> tensor. `sample` as for scatter_scanline.
  void scatter_tile(const void *src, const glm::ivec2 org, const int sample = -1);

  //! Set the raster layout tags on an output file.
  void set_raster_tags(TIFF *out);

  //! Gather the tensor into an output file, scanline by scanline.
  bool write_raster(TIFF *out);

  template<typename T>
  void scatter_scanline_t(const T *src, const size_t row, const int sample);

  template<typename T>
  void gather_scanline_t(T *dst, const size_t row);

  template<typename T>
  void scatter_tile_t(const T *src, const glm::ivec2 org, const int sample);

  //! Bytes per stored element in the tensor, which is what a staging buffer is
  //! sized from.
  inline size_t elem_size() const { return (this->_bits == 64) ? 8 : 4; }

  //! Bytes in one output scanline, which holds every sample of every column.
  inline size_t scanline_size() const {
    return (size_t)this->_width * this->_samples * this->elem_size();
  }

  bool meta_loaded = false; //!< Flag: Is Meta-Data Loaded
  bool tiled_image = false; //!< Flag: Is Image Tiled

  std::string filename; //!< Original Filename

  uint32_t _width = 0;   //!< Image Width
  uint32_t _height = 0;  //!< Image Height
  uint32_t _bits = 0;    //!< Pixel Bit-Depth
  uint32_t _samples = 1; //!< Samples per Pixel

  uint32_t _twidth = 0;  //!< Tile Width
  uint32_t _theight = 0; //!< Tile Height

  uint16_t _planar = PLANARCONFIG_CONTIG; //!< Sample Storage Organization

  silt::shape _shape;   //!< Underlying Data Shape
  silt::tensor _tensor; //!< Underlying Data Tensor
};

//! Scatter one file scanline into the tensor, reversing y.
template<typename T>
void tiff::scatter_scanline_t(const T *src, const size_t row, const int sample) {

  T *data = (T *)this->_tensor.data();

  if (sample < 0) {
    for (size_t x = 0; x < this->width(); ++x)
      for (size_t s = 0; s < this->samples(); ++s)
        data[this->tensor_index(x, row, s)] = src[x * this->samples() + s];
  }

  else {
    for (size_t x = 0; x < this->width(); ++x)
      data[this->tensor_index(x, row, (size_t)sample)] = src[x];
  }
}

inline void tiff::scatter_scanline(const void *src, const size_t row, const int sample) {
  if (this->bits() == 32)
    this->scatter_scanline_t((const float *)src, row, sample);
  if (this->bits() == 64)
    this->scatter_scanline_t((const double *)src, row, sample);
}

//! Gather one file scanline out of the tensor, reversing y.
template<typename T>
void tiff::gather_scanline_t(T *dst, const size_t row) {

  const T *data = (const T *)this->_tensor.data();

  for (size_t x = 0; x < this->width(); ++x)
    for (size_t s = 0; s < this->samples(); ++s)
      dst[x * this->samples() + s] = data[this->tensor_index(x, row, s)];
}

inline void tiff::gather_scanline(void *dst, const size_t row) {
  if (this->bits() == 32)
    this->gather_scanline_t((float *)dst, row);
  if (this->bits() == 64)
    this->gather_scanline_t((double *)dst, row);
}

//! Scatter one file tile into the tensor, reversing y.
template<typename T>
void tiff::scatter_tile_t(const T *src, const glm::ivec2 org, const int sample) {

  T *data = (T *)this->_tensor.data();

  // An interleaved tile holds every sample; a plane holds exactly one.
  const size_t stride = (sample < 0) ? this->samples() : 1;

  for (size_t ix = 0; ix < this->_twidth; ix++)
    for (size_t iy = 0; iy < this->_theight; iy++) {

      const glm::ivec2 fpos = org + glm::ivec2(ix, iy);

      if (fpos.x >= this->width())
        continue;
      if (fpos.y >= this->height())
        continue;

      for (size_t s = 0; s < stride; ++s) {
        const size_t c = (sample < 0) ? s : (size_t)sample;
        data[this->tensor_index(fpos.x, fpos.y, c)] = src[(iy * this->_twidth + ix) * stride + s];
      }
    }
}

inline void tiff::scatter_tile(const void *src, const glm::ivec2 org, const int sample) {
  if (this->bits() == 32)
    this->scatter_tile_t((const float *)src, org, sample);
  if (this->bits() == 64)
    this->scatter_tile_t((const double *)src, org, sample);
}

//! Set the raster layout tags on an output file.
inline void tiff::set_raster_tags(TIFF *out) {

  TIFFSetField(out, TIFFTAG_IMAGEWIDTH, this->width());
  TIFFSetField(out, TIFFTAG_IMAGELENGTH, this->height());
  TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, this->samples());
  TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, this->bits());
  TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, this->width()));

  // PHOTOMETRIC_MINISBLACK accounts for one sample, so the remaining channels
  // are declared extra. Unspecified, not alpha: they are physical quantities,
  // and a reader must not premultiply them.
  if (this->samples() > 1) {
    std::vector<uint16_t> extra(this->samples() - 1, EXTRASAMPLE_UNSPECIFIED);
    TIFFSetField(out, TIFFTAG_EXTRASAMPLES, (uint16_t)extra.size(), extra.data());
  }
}

//! Gather the tensor into an output file, scanline by scanline.
inline bool tiff::write_raster(TIFF *out) {

  // Sized from the stored element, not bits()/8: see elem_size.
  std::vector<uint8_t> nbuf(this->scanline_size());

  for (uint32_t row = 0; row < this->height(); row++) {
    this->gather_scanline(nbuf.data(), row);
    if (TIFFWriteScanline(out, nbuf.data(), row, 0) < 0)
      return false;
  }

  return true;
}

//! Load TIFF Metadata
inline bool tiff::peek(const char *filename) {

  TIFF *tif = TIFFOpen(filename, "r");
  if (tif == NULL) {
    throw silt::error::missing_file(filename);
    return false;
  }

  if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &this->_width))
    return false;

  if (!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &this->_height))
    return false;

  // BITSPERSAMPLE, SAMPLESPERPIXEL and PLANARCONFIG are TIFF_SHORT fields:
  // TIFFGetField writes 16 bits, so it must be handed a uint16_t.
  uint16_t bits = 0;
  if (!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits))
    return false;
  this->_bits = bits;

  // Both default to their specified values when the tag is absent.
  uint16_t samples = 1;
  if (!TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples))
    samples = 1;
  this->_samples = samples;

  uint16_t planar = PLANARCONFIG_CONTIG;
  if (!TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planar))
    planar = PLANARCONFIG_CONTIG;
  this->_planar = planar;

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
inline bool tiff::read(const char *filename) {

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

  // Shape is (W, H) for one sample and (W, H, C) beyond: dim0 = x (east),
  // dim1 = y (north), channel contiguous. The file is row-major with row 0
  // north, so the scanline helpers reverse y on the way in and out. That
  // conversion happens here and nowhere else.

  if (this->samples() == 1)
    this->_shape = silt::shape(this->width(), this->height());
  else
    this->_shape = silt::shape(this->width(), this->height(), this->samples());

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

    // libtiff sizes a scanline, accounting for samples per pixel and for the
    // planar configuration. Computing it here instead would overrun the buffer
    // on any file with more than one sample.
    const tmsize_t size = TIFFScanlineSize(tif);
    if (size <= 0) {
      TIFFClose(tif);
      throw std::runtime_error("TIFF scanline size unavailable (malformed directory)");
    }

    std::vector<uint8_t> nbuf(size);

    if (this->_planar == PLANARCONFIG_SEPARATE) {
      for (size_t s = 0; s < this->samples(); ++s)
        for (size_t row = 0; row < this->height(); row++) {
          TIFFReadScanline(tif, nbuf.data(), row, (uint16_t)s);
          this->scatter_scanline(nbuf.data(), row, (int)s);
        }
    }

    else {
      for (size_t row = 0; row < this->height(); row++) {
        TIFFReadScanline(tif, nbuf.data(), row);
        this->scatter_scanline(nbuf.data(), row);
      }
    }
  }

  else {

    const size_t nwidth = (this->width() + this->_twidth - 1) / this->_twidth;
    const size_t nheight = (this->height() + this->_theight - 1) / this->_theight;

    // One tile per plane when separate, one tile for every sample otherwise.
    const size_t planes = (this->_planar == PLANARCONFIG_SEPARATE) ? this->samples() : 1;

    const tmsize_t size = TIFFTileSize(tif);
    if (size <= 0) {
      TIFFClose(tif);
      throw std::runtime_error("TIFF tile size unavailable (malformed directory)");
    }

    std::vector<uint8_t> nbuf(size);

    for (size_t plane = 0; plane < planes; ++plane)
      for (size_t nx = 0; nx < nwidth; ++nx)
        for (size_t ny = 0; ny < nheight; ++ny) {

          const glm::ivec2 npos(nx, ny);
          const glm::ivec2 norg = npos * glm::ivec2(this->_twidth, this->_theight);

          if (!TIFFReadTile(tif, nbuf.data(), norg.x, norg.y, 0, (uint16_t)plane)) {
            continue;
          }

          // Reverses y, as the scanline helpers do.
          this->scatter_tile(nbuf.data(), norg, (planes > 1) ? (int)plane : -1);
        }
  }

  TIFFClose(tif);
  return true;
}

inline bool tiff::write(const char *filename) {

  TIFF *out = TIFFOpen(filename, "w");

  this->set_raster_tags(out);
  this->write_raster(out);

  TIFFClose(out);
  return true;
}

}; // end of namespace io
}; // end of namespace soil
