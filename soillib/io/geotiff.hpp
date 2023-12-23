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
  { TIFFTAG_GEOPIXELSCALE,  -1,-1, TIFF_DOUBLE, FIELD_CUSTOM,
    true, true,   "GeoPixelScale" },
  { TIFFTAG_INTERGRAPH_MATRIX,-1,-1, TIFF_DOUBLE, FIELD_CUSTOM,
    true, true,   "Intergraph TransformationMatrix" },
  { TIFFTAG_GEOTRANSMATRIX, -1,-1, TIFF_DOUBLE, FIELD_CUSTOM,
    true, true,   "GeoTransformationMatrix" },
  { TIFFTAG_GEOTIEPOINTS, -1,-1, TIFF_DOUBLE, FIELD_CUSTOM,
    true, true,   "GeoTiePoints" },
  { TIFFTAG_GEOKEYDIRECTORY,-1,-1, TIFF_SHORT,  FIELD_CUSTOM,
    true, true,   "GeoKeyDirectory" },
  { TIFFTAG_GEODOUBLEPARAMS,  -1,-1, TIFF_DOUBLE, FIELD_CUSTOM,
    true, true,   "GeoDoubleParams" },
  { TIFFTAG_GEOASCIIPARAMS, -1,-1, TIFF_ASCII, FIELD_CUSTOM,
    true, false,  "GeoASCIIParams" },
  { TIFFTAG_GDAL_METADATA, -1,-1, TIFF_ASCII, FIELD_CUSTOM,
    true, false,  "GeoGDALMetaData" },
  { TIFFTAG_GDAL_NODATA, -1,-1, TIFF_ASCII, FIELD_CUSTOM,
    true, false,  "GeoGDALNoData" },    
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

// TIFF Implementation

template<typename T = float>
struct geotiff: soil::io::tiff<T> {

  using soil::io::tiff<T>::allocate;
  using soil::io::tiff<T>::operator[];

  geotiff(){};
  geotiff(const char* filename){ read(filename); };
  geotiff(const size_t width, const size_t height):img<T>(width, height){}
  geotiff(const glm::ivec2 res):img<T>(res){}

  bool meta(const char* filename);
  bool read(const char* filename);
  bool write(const char* filename);

  glm::dvec3 scale;
  glm::dvec3 coords[2];

  // Tiling Handling
  bool tiled = false;
  glm::tvec2<uint32_t> tiledim;
  glm::tvec2<uint32_t> tilenum;

};

// Implementations

template<typename T>
bool geotiff<T>::meta(const char* filename){

  _XTIFFInitialize();

  TIFF* tif = TIFFOpen(filename, "r");

  uint32_t width;
  uint32_t height;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

  uint32_t twidth = 0;
  uint32_t theight = 0;

  if(TIFFGetField(tif, TIFFTAG_TILEWIDTH, &twidth) 
  && TIFFGetField(tif, TIFFTAG_TILELENGTH, &theight)){
    tiled = true;
    tiledim = glm::tvec2<uint32_t>(twidth, theight);
    tilenum = (glm::tvec2<uint32_t>(width, height) - glm::tvec2<uint32_t>(1))/tiledim + glm::tvec2<uint32_t>(1);
  }

  // Read Meta-Data

  int count;
  double* values;
  TIFFGetField(tif, TIFFTAG_GEOPIXELSCALE, &count, &values);
  scale.x = values[0];
  scale.y = values[1];
  scale.z = values[2];

  TIFFGetField(tif, TIFFTAG_GEOTIEPOINTS, &count, &values);
  coords[0].x = values[0];
  coords[0].y = values[1];
  coords[0].z = values[2];
  coords[1].x = values[3];
  coords[1].y = values[4];
  coords[1].z = values[5];

  TIFFClose(tif);
  return true;

}

template<typename T>
bool geotiff<T>::read(const char* filename){

  _XTIFFInitialize();

  TIFF* tif = TIFFOpen(filename, "r");

  uint32_t width;
  uint32_t height;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

  if(width != this->width || height != this->height || this->data == NULL){
    this->width = width;
    this->height = height;
    if(this->data != NULL){
      delete[] this->data;
      this->data = NULL;
    }
    allocate();
  }

  if(!tiled){
    T* buf = this->data;
    for (size_t row = 0; row < this->height; row++){
      TIFFReadScanline(tif, buf, row);
      buf += this->width;
    }
  } else {

    T* buf = this->data;
    T* nbuf = new T[this->tiledim.x*this->tiledim.y];

    for(size_t tx = 0; tx < this->width; tx += this->tiledim.x)
    for(size_t ty = 0; ty < this->height; ty += this->tiledim.y){
      
      if(!TIFFReadTile(tif, nbuf, tx, ty, 0, 0)){
        std::cout<<"FAILED TO READ TILE"<<std::endl;
      }

      glm::ivec2 ipos = glm::ivec2(tx, ty);

      for(size_t ix = 0; ix < this->tiledim.x; ix++)
      for(size_t iy = 0; iy < this->tiledim.y; iy++){

        glm::ivec2 tpos = glm::ivec2(ix, iy);
        glm::ivec2 fpos = ipos + tpos;

        if(fpos.x >= this->width) continue;
        if(fpos.y >= this->height) continue;

        buf[fpos.y * this->width + fpos.x] = nbuf[iy * this->tiledim.x + ix];
      }

    }
    delete[] nbuf;
  }

  TIFFClose(tif);
  return true;

}

template<typename T>
bool geotiff<T>::write(const char *filename) {
  
  TIFF *out= TIFFOpen(filename, "w");
  TIFFSetField(out, TIFFTAG_IMAGEWIDTH, this->width);
  TIFFSetField(out, TIFFTAG_IMAGELENGTH, this->height);
  TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
  TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8*sizeof(T));    // set the size of the channels
  TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
  //   Some other essential fields to set that you do not have to understand for now.
  TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  // We set the strip size of the file to be size of one row of pixels
  TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, this->width));

  T* buf = this->data;
  //Now writing image to the file one strip at a time
  for (uint32_t row = 0; row < this->height; row++) {
      if (TIFFWriteScanline(out, buf, row, 0) < 0)
      break;
    buf += this->width;
  }

  TIFFClose(out);
  return true; 
}

}; // end of namespace io
}; // end of namespace soil

#endif