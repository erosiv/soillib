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

// GeoTIFF Implementation

template<typename T = float>
struct geotiff: soil::io::tiff<T> {

  using soil::io::tiff<T>::tiff;
  using soil::io::tiff<T>::allocate;
  using soil::io::tiff<T>::operator[];

  geotiff(){};
  geotiff(const char* filename){ meta(filename); read(filename); };
  geotiff(const size_t width, const size_t height):img<T>(width, height){}
  geotiff(const glm::ivec2 res):img<T>(res){}

  bool meta(const char* filename);
  bool read(const char* filename);
  bool write(const char* filename);

  // Projection

  inline glm::vec2 scale() const { return glm::vec2(_scale); }
  inline glm::vec2 dim() const   { return glm::vec2(this->width, this->height); }
  inline glm::vec2 min() const   { return glm::vec2(_coords[1]); }
  inline glm::vec2 max() const   { return min() + scale()*dim(); }
  inline glm::vec2 map(const glm::vec2 p) const { return min() + scale()*p; }

private:
  glm::vec3 _scale{1};
  glm::vec3 _coords[2]{glm::vec3{0}, glm::vec3{0}};
};

// Implementations

template<typename T>
bool geotiff<T>::meta(const char* filename){

  _XTIFFInitialize();

  if(!tiff<T>::meta(filename))
    return false;

  TIFF* tif = TIFFOpen(filename, "r");

  int count = 0;
  char* text_ptr = NULL;
  if(TIFFGetField(tif, TIFFTAG_GDAL_NODATA, &text_ptr)){}

  // Read Meta-Data

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

template<typename T>
bool geotiff<T>::read(const char* filename){
  _XTIFFInitialize();
  return tiff<T>::read(filename);
}

template<typename T>
bool geotiff<T>::write(const char *filename) {
  return tiff<T>::write(filename);
}

}; // end of namespace io
}; // end of namespace soil

#endif