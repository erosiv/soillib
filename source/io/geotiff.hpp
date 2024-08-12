#ifndef SOILLIB_IO_GEOTIFF
#define SOILLIB_IO_GEOTIFF

#include <soillib/io/tiff.hpp>

namespace soil {
namespace io {

// GeoTIFF Specification w. GDAL Extension

#define N(a) (sizeof (a) / sizeof (a[0]))
#define TIFFTAG_GEOPIXELSCALE         33550
#define TIFFTAG_INTERGRAPH_MATRIX     33920
#define TIFFTAG_GEOTIEPOINTS          33922
#define TIFFTAG_GEOTRANSMATRIX        34264
#define TIFFTAG_GEOKEYDIRECTORY       34735
#define TIFFTAG_GEODOUBLEPARAMS       34736
#define TIFFTAG_GEOASCIIPARAMS        34737
#define TIFFTAG_GDAL_METADATA         42112
#define TIFFTAG_GDAL_NODATA           42113

static const TIFFFieldInfo xtiffFieldInfo[] = {
  { TIFFTAG_GEOPIXELSCALE,      -1, -1, TIFF_DOUBLE,  FIELD_CUSTOM, true, true,   (char*)"GeoPixelScale" },
  { TIFFTAG_INTERGRAPH_MATRIX,  -1, -1, TIFF_DOUBLE,  FIELD_CUSTOM, true, true,   (char*)"Intergraph TransformationMatrix" },
  { TIFFTAG_GEOTRANSMATRIX,     -1, -1, TIFF_DOUBLE,  FIELD_CUSTOM, true, true,   (char*)"GeoTransformationMatrix" },
  { TIFFTAG_GEOTIEPOINTS,       -1, -1, TIFF_DOUBLE,  FIELD_CUSTOM, true, true,   (char*)"GeoTiePoints" },
  { TIFFTAG_GEOKEYDIRECTORY,    -1, -1, TIFF_SHORT,   FIELD_CUSTOM, true, true,   (char*)"GeoKeyDirectory" },
  { TIFFTAG_GEODOUBLEPARAMS,    -1, -1, TIFF_DOUBLE,  FIELD_CUSTOM, true, true,   (char*)"GeoDoubleParams" },
  { TIFFTAG_GEOASCIIPARAMS,     -1, -1, TIFF_ASCII,   FIELD_CUSTOM, true, false,  (char*)"GeoASCIIParams" },
  { TIFFTAG_GDAL_METADATA,      -1, -1, TIFF_ASCII,   FIELD_CUSTOM, true, false,  (char*)"GDAL_METADATA" },
  { TIFFTAG_GDAL_NODATA,        -1, -1, TIFF_ASCII,   FIELD_CUSTOM, true, false,  (char*)"GDAL_NODATA" },    
};

// Custom Tiff-Tag Handling Initializer Hook

static TIFFExtendProc _ParentExtender = NULL;

static void _XTIFFDefaultDirectory(TIFF *tif){
  TIFFMergeFieldInfo(tif, xtiffFieldInfo, N(xtiffFieldInfo));
  if (_ParentExtender)
      (*_ParentExtender)(tif);
}

static void _XTIFFInitialize(void) {
  static int first_time=1;
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
//template<typename T = float>
struct geotiff: soil::io::tiff {

  using soil::io::tiff::tiff;
  using soil::io::tiff::width;
  using soil::io::tiff::height;
  
  //using soil::io::tiff<T>::allocate;
  //using soil::io::tiff<T>::operator[];

  geotiff(){};
  geotiff(const char* filename){ meta(filename); read(filename); };
  //geotiff(const size_t width, const size_t height):img<T>(width, height){}
  //geotiff(const glm::ivec2 res):img<T>(res){}

  bool meta(const char* filename);
  bool read(const char* filename);
  bool write(const char* filename);

  // Projection

//  inline std::array<float, 2> max() const { return _coords}

  inline glm::vec2 scale() const { return glm::vec2(_scale.x, _scale.y); }
  inline glm::vec2 dim() const   { return glm::vec2(this->width(), this->height()); }
  inline glm::vec2 min() const   { return glm::min(glm::vec2(_coords[1]), glm::vec2(_coords[1]) + glm::vec2(_scale.x, _scale.y)*dim()); }
  inline glm::vec2 max() const   { return glm::max(glm::vec2(_coords[1]), glm::vec2(_coords[1]) + glm::vec2(_scale.x, _scale.y)*dim()); }
  // Vienna DGM: Requires User-Facing Ability to Alter Scle Cleanly...
  //inline glm::vec2 min() const   { return glm::min(glm::vec2(_coords[1]), glm::vec2(_coords[1]) + glm::vec2(_scale.x, -_scale.y)*dim()); }
  //inline glm::vec2 max() const   { return glm::max(glm::vec2(_coords[1]), glm::vec2(_coords[1]) + glm::vec2(_scale.x, -_scale.y)*dim()); }
  inline glm::vec2 map(const glm::vec2 p) const { return min() + scale()*p; }

private:
  
  void setNaN();  //!< Set Available NoData Values to NaN=

  std::string nodata = "";
  glm::vec3 _scale{1};
  glm::vec3 _coords[2]{glm::vec3{0}, glm::vec3{0}};
};

// Implementations

//template<typename T>
bool geotiff::meta(const char* filename){

  _XTIFFInitialize();

  if(!tiff::meta(filename))
    return false;

  TIFF* tif = TIFFOpen(filename, "r");

  char* text_ptr;
  if(TIFFGetField(tif, TIFFTAG_GDAL_NODATA, &text_ptr)){
    this->nodata = std::string(text_ptr);
  }

  // Read Meta-Data

  int count = 0;
  double* values;
  if(TIFFGetField(tif, TIFFTAG_GEOPIXELSCALE, &count, &values)){
    _scale.x = values[0];
    _scale.y = values[1];
    _scale.z = values[2];
  }

  if(TIFFGetField(tif, TIFFTAG_GEOTIEPOINTS, &count, &values)){
    _coords[0].x = values[0];
    _coords[0].y = values[1];
    _coords[0].z = values[2];
    _coords[1].x = values[3];
    _coords[1].y = values[4];
    _coords[1].z = values[5];
  }

  TIFFClose(tif);
  return true;

}

bool geotiff::read(const char* filename){
  
  geotiff::meta(filename);
  if(!tiff::read(filename))
    return false;

  geotiff::setNaN();
  return true;
}

bool geotiff::write(const char *filename) {
  return tiff::write(filename);
}

void geotiff::setNaN(){

  if(this->bits() == 32){
    auto nan = std::numeric_limits<float>::quiet_NaN();
    auto array = std::get<soil::array_t<float>>(this->_array._array);
    const float _nodata = std::stof(this->nodata);
    for(size_t i = 0; i < array.elem(); ++i){
      if(array[i] == _nodata) 
        array[i] = nan;
    }
  }

  if(this->bits() == 64){
    auto nan = std::numeric_limits<double>::quiet_NaN();
    auto array = std::get<soil::array_t<double>>(this->_array._array);
    const double _nodata = std::stof(this->nodata);
    for(size_t i = 0; i < array.elem(); ++i){
      if(array[i] == _nodata) 
        array[i] = nan;
    }
  }

} 

}; // end of namespace io
}; // end of namespace soil

#endif