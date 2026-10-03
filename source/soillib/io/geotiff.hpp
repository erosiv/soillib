#pragma once

#include <soillib/soillib.hpp>

#include <soillib/io/tiff.hpp>

namespace soil {
namespace io {

// GeoTIFF Specification w. GDAL Extension

#define N(a) (sizeof(a) / sizeof(a[0]))
#define TIFFTAG_GEOPIXELSCALE 33550
#define TIFFTAG_INTERGRAPH_MATRIX 33920
#define TIFFTAG_GEOTIEPOINTS 33922
#define TIFFTAG_GEOTRANSMATRIX 34264
#define TIFFTAG_GEOKEYDIRECTORY 34735
#define TIFFTAG_GEODOUBLEPARAMS 34736
#define TIFFTAG_GEOASCIIPARAMS 34737
#define TIFFTAG_GDAL_METADATA 42112
#define TIFFTAG_GDAL_NODATA 42113

static const TIFFFieldInfo xtiffFieldInfo[] = {
    {TIFFTAG_GEOPIXELSCALE, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, true, true, (char *)"GeoPixelScale"},
    {TIFFTAG_GEOTIEPOINTS, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, true, true, (char *)"GeoTiePoints"},
    {TIFFTAG_INTERGRAPH_MATRIX, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, true, true, (char *)"Intergraph TransformationMatrix"},
    {TIFFTAG_GEOTRANSMATRIX, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, true, true, (char *)"GeoTransformationMatrix"},
    {TIFFTAG_GEODOUBLEPARAMS, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, true, true, (char *)"GeoDoubleParams"},
    {TIFFTAG_GEOASCIIPARAMS, -1, -1, TIFF_ASCII, FIELD_CUSTOM, true, false, (char *)"GeoASCIIParams"},
    {TIFFTAG_GDAL_METADATA, -1, -1, TIFF_ASCII, FIELD_CUSTOM, true, false, (char *)"GDAL_METADATA"},
    {TIFFTAG_GDAL_NODATA, -1, -1, TIFF_ASCII, FIELD_CUSTOM, true, false, (char *)"GDAL_NODATA"},
    {TIFFTAG_GEOKEYDIRECTORY, -1, -1, TIFF_SHORT, FIELD_CUSTOM, true, true, (char *)"GeoKeyDirectory"},
};

// Custom Tiff-Tag Handling Initializer Hook

static TIFFExtendProc _ParentExtender = NULL;

static void _XTIFFDefaultDirectory(TIFF *tif) {
  TIFFMergeFieldInfo(tif, xtiffFieldInfo, N(xtiffFieldInfo));
  if (_ParentExtender)
    (*_ParentExtender)(tif);
}

static void _XTIFFInitialize(void) {
  static int first_time = 1;
  if (!first_time)
    return;
  first_time = 0;
  _ParentExtender = TIFFSetTagExtender(_XTIFFDefaultDirectory);
}

//! geotiff<T> is a generic, strict-typed .tiff interface
//! for loading and saving images w. GeoTIFF information.
//!
//! File meta-data such as size, scale and position can be
//! loaded before the raw image data is loaded for better
//! optimization. geotiff<T> also implements GDAL meta-data.
//!
//! GeoTIFF implements geo-spatial meta functions, e.g.
//! map to world-space or min and max extent in world-space
//! for convenience.
//!
// template<typename T = float>
struct geotiff: soil::io::tiff {

  using soil::io::tiff::height;
  using soil::io::tiff::tiff;
  using soil::io::tiff::width;

  geotiff() {};
  geotiff(const silt::tensor &_tensor): tiff(_tensor) {
    // Mirror the raster dimensions into the metadata; set_min needs them.
    // The tie point keeps its default until a caller georeferences the image.
    this->_meta.width = this->width();
    this->_meta.height = this->height();
    this->_meta.bits = this->bits();
  }

  geotiff(const char *filename) {
    peek(filename);
    read(filename);
  };

  bool peek(const char *filename);
  bool read(const char *filename);
  bool write(const char *filename);

  //! GeoTIFF Metadata Type
  struct meta_t {
    std::string filename;
    size_t width;
    size_t height;
    size_t bits;

    std::string gdal_nodata;
    std::string gdal_metadata;
    std::string geoasciiparams;

    std::vector<double> scale = {1.0, 1.0, 1.0};
    std::vector<double> coords = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> params;
    std::vector<short> keydir;

    inline glm::vec2 dim() const { return glm::vec2(width, height); }

    //! World position of the raster tie point: GeoTIFF ModelTiepointTag gives
    //! the NORTH-WEST corner, i.e. raster (i, j) = (0, 0), which is row 0.
    inline glm::vec2 tiepoint() const { return glm::vec2(coords[3], coords[4]); }

    //! Cell size, always positive. ModelPixelScale is specified positive with
    //! the row index increasing southward; some writers emit a negative y
    //! regardless, so the magnitude is taken and the direction applied below.
    inline glm::vec2 cell() const { return glm::abs(glm::vec2(scale[0], scale[1])); }

    //! South-west corner: tensor cell (0, 0). World y decreases down the
    //! raster, so the tie point's y is the maximum, not the minimum.
    inline glm::vec2 min() const { return glm::vec2(tiepoint().x, tiepoint().y - cell().y * dim().y); }

    //! North-east corner.
    inline glm::vec2 max() const { return glm::vec2(tiepoint().x + cell().x * dim().x, tiepoint().y); }

    //! Georeference from the south-west corner, converting to the north-west
    //! tie point the format stores. `scale` and `height` must already be set,
    //! so assign the scale before the origin.
    inline void set_min(const glm::vec2 m) {
      coords[3] = m.x;
      coords[4] = double(m.y) + double(cell().y) * double(height);
    }
  };

  // Projection

  //! Cell size in world units, always positive. See meta_t::cell.
  inline glm::vec2 scale() const { return _meta.cell(); }
  inline glm::vec2 dim() const { return glm::vec2(this->width(), this->height()); }

  //! South-west corner (tensor cell (0, 0)) and north-east corner.
  inline glm::vec2 min() const { return _meta.min(); }
  inline glm::vec2 max() const { return _meta.max(); }

  //! Map a tensor coordinate (x east, y north) into world space.
  inline glm::vec2 map(const glm::vec2 p) const { return min() + scale() * p; }

  // Basic Get and Copy
  // The idea is for us to store a single one...
  // Then just copy the essential stuff!

  void unsetNaN(); //!< Set Available NoData Values to NaN=
  meta_t _meta;    //!< Local Meta-Data

private:
  void setNaN(); //!< Set Available NoData Values to NaN=
};

// Implementations

bool geotiff::peek(const char *filename) {

  _XTIFFInitialize();

  if (!tiff::peek(filename))
    return false;

  this->_meta.filename = this->filename;
  this->_meta.width = this->width();
  this->_meta.height = this->height();
  this->_meta.bits = this->bits();

  TIFF *tif = TIFFOpen(filename, "r");
  if (tif == NULL) {
    throw silt::error::missing_file(filename);
    return false;
  }

  // Load Meta-Data

  int count = 0;
  char *text_ptr;
  double *values;
  short *short_data;

  if (TIFFGetField(tif, TIFFTAG_GDAL_NODATA, &text_ptr))
    this->_meta.gdal_nodata = std::string(text_ptr);

  if (TIFFGetField(tif, TIFFTAG_GDAL_METADATA, &text_ptr))
    this->_meta.gdal_metadata = std::string(text_ptr);

  if (TIFFGetField(tif, TIFFTAG_GEOASCIIPARAMS, &text_ptr))
    this->_meta.geoasciiparams = std::string(text_ptr);

  if (TIFFGetField(tif, TIFFTAG_GEOPIXELSCALE, &count, &values)){
    this->_meta.scale = std::vector<double>(values, values + count);
    if(this->_meta.scale[2] == 0.0f)
      this->_meta.scale[2] = 1.0f;
  }

  if (TIFFGetField(tif, TIFFTAG_GEOTIEPOINTS, &count, &values))
    this->_meta.coords = std::vector<double>(values, values + count);

  if (TIFFGetField(tif, TIFFTAG_GEODOUBLEPARAMS, &count, &values))
    this->_meta.params = std::vector<double>(values, values + count);

  if (TIFFGetField(tif, TIFFTAG_GEOKEYDIRECTORY, &count, &short_data))
    this->_meta.keydir = std::vector<short>(short_data, short_data + count);

  TIFFClose(tif);
  return true;
}

bool geotiff::read(const char *filename) {

  geotiff::peek(filename);
  if (!tiff::read(filename))
    return false;

  geotiff::setNaN();
  return true;
}

bool geotiff::write(const char *filename) {

  _XTIFFInitialize();

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

  // GDAL Tags

  if (!_meta.scale.empty())
    TIFFSetField(out, TIFFTAG_GEOPIXELSCALE, _meta.scale.size(), &_meta.scale[0]);
  if (!_meta.coords.empty())
    TIFFSetField(out, TIFFTAG_GEOTIEPOINTS, _meta.coords.size(), &_meta.coords[0]);
  if (!_meta.params.empty())
    TIFFSetField(out, TIFFTAG_GEODOUBLEPARAMS, _meta.params.size(), &_meta.params[0]);
  if (!_meta.keydir.empty())
    TIFFSetField(out, TIFFTAG_GEOKEYDIRECTORY, _meta.keydir.size(), &_meta.keydir[0]);

  if (!_meta.gdal_nodata.empty())
    TIFFSetField(out, TIFFTAG_GDAL_NODATA, _meta.gdal_nodata.c_str());
  if (!_meta.gdal_metadata.empty())
    TIFFSetField(out, TIFFTAG_GDAL_METADATA, _meta.gdal_metadata.c_str());
  if (!_meta.geoasciiparams.empty())
    TIFFSetField(out, TIFFTAG_GEOASCIIPARAMS, _meta.geoasciiparams.c_str());

  // Output Data: scanlines are gathered with y reversed, as in tiff::write.

  // Sized from the stored element, not bits()/8: see tiff::elem_size.
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

void geotiff::setNaN() {

  if (this->_meta.gdal_nodata == "")
    return;

  if (this->bits() == 16) {
    auto nan = std::numeric_limits<float>::quiet_NaN();
    auto buffer = this->_tensor.as<float>();
    const float _nodata = std::stof(this->_meta.gdal_nodata);
    for (size_t i = 0; i < buffer.elem(); ++i) {
      if (buffer[i] == _nodata)
        buffer[i] = nan;
    }
  }

  if (this->bits() == 32) {
    auto nan = std::numeric_limits<float>::quiet_NaN();
    auto buffer = this->_tensor.as<float>();
    const float _nodata = std::stof(this->_meta.gdal_nodata);
    for (size_t i = 0; i < buffer.elem(); ++i) {
      if (buffer[i] == _nodata)
        buffer[i] = nan;
    }
  }

  if (this->bits() == 64) {
    auto nan = std::numeric_limits<double>::quiet_NaN();
    auto buffer = this->_tensor.as<double>();
    const double _nodata = std::stod(this->_meta.gdal_nodata);
    for (size_t i = 0; i < buffer.elem(); ++i) {
      if (buffer[i] == _nodata)
        buffer[i] = nan;
    }
  }
}

void geotiff::unsetNaN() {

  if (this->_meta.gdal_nodata == "")
    return;

  if (this->bits() == 16) {
    auto nan = std::numeric_limits<float>::quiet_NaN();
    auto buffer = this->_tensor.as<float>();
    const float _nodata = std::stof(this->_meta.gdal_nodata);
    for (size_t i = 0; i < buffer.elem(); ++i) {
      if (buffer[i] == nan) {
        buffer[i] = _nodata;
      }
    }
  }

  if (this->bits() == 32) {
    auto nan = std::numeric_limits<float>::quiet_NaN();
    auto buffer = this->_tensor.as<float>();
    const float _nodata = std::stof(this->_meta.gdal_nodata);
    for (size_t i = 0; i < buffer.elem(); ++i) {
      if (buffer[i] == nan)
        buffer[i] = _nodata;
    }
  }

  if (this->bits() == 64) {
    auto nan = std::numeric_limits<double>::quiet_NaN();
    auto buffer = this->_tensor.as<double>();
    const double _nodata = std::stod(this->_meta.gdal_nodata);
    for (size_t i = 0; i < buffer.elem(); ++i) {
      if (buffer[i] == nan)
        buffer[i] = _nodata;
    }
  }
}

}; // end of namespace io
}; // end of namespace soil
