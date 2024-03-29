#ifndef SOILLIB_IO_TIFF
#define SOILLIB_IO_TIFF

#include <soillib/io/img.hpp>
#include <tiffio.h>

namespace soil {
namespace io {

//! tiff<T> is a generic, strict-typed .tiff interface
//! for reading and writing generic image data to disk.
//!
//! tiff<T> supports multiple different bit-depths, as
//! well as the reading of tiled .tiff images.
//!
template<typename T = float>
struct tiff: soil::io::img<T> {

  using soil::io::img<T>::value_t;
  using soil::io::img<T>::img;
  using soil::io::img<T>::allocate;
  using soil::io::img<T>::operator[];

  tiff(const char* filename){ read(filename); };

  bool meta(const char* filename);  //!< Load TIFF Metadata
  bool read(const char* filename);  //!< Read TIFF Raw Data
  bool write(const char* filename); //!< Write TIFF Raw Data

private:
  bool meta_loaded = false;
  bool tiled_image = false;
  uint32_t twidth = 0;
  uint32_t theight = 0;
};

//! Load TIFF Metadata
template<typename T>
bool tiff<T>::meta(const char* filename){

  TIFF* tif = TIFFOpen(filename, "r");

  // Meta-Data

  if(!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &this->width))
    return false;

  if(!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &this->height))
    return false;

  if(TIFFGetField(tif, TIFFTAG_TILEWIDTH, &this->twidth))
    this->tiled_image = true;

  if(TIFFGetField(tif, TIFFTAG_TILELENGTH, &this->theight))
    this->tiled_image = true;

  TIFFClose(tif);

  this->meta_loaded = true;
  return true;

}

//! Read TIFF Raw Data
template<typename T>
bool tiff<T>::read(const char* filename){

  if(!meta_loaded){
    this->meta(filename);
  }

  TIFF* tif = TIFFOpen(filename, "r");

  if(this->data != NULL){
    delete[] this->data;
    this->data = NULL;
  }
  allocate();

  // Load Tiled / Non-Tiled Images

  if(!this->tiled_image){

    T* buf = this->data;
    for (size_t row = 0; row < this->height; row++){
      TIFFReadScanline(tif, buf, row);
      buf += this->width;
    }

  } else {

    const size_t tsize = this->twidth * this->theight;
    const size_t nwidth = (this->width + this->twidth - 1)/this->twidth;
    const size_t nheight = (this->height + this->theight - 1)/this->theight;

    T* buf = this->data;
    T* nbuf = new T[tsize];

    for(size_t nx = 0; nx < nwidth; ++nx)
    for(size_t ny = 0; ny < nheight; ++ny){

      const glm::ivec2 npos(nx, ny);
      const glm::ivec2 norg = npos*glm::ivec2(this->twidth, this->theight);

      if(!TIFFReadTile(tif, nbuf, norg.x, norg.y, 0, 0)){
        continue;
      }

      for(size_t ix = 0; ix < this->twidth; ix++)
      for(size_t iy = 0; iy < this->theight; iy++){

        glm::ivec2 tpos = glm::ivec2(ix, iy);
        glm::ivec2 fpos = norg + tpos;

       if(fpos.x >= this->width) continue;
       if(fpos.y >= this->height) continue;

        if(buf[fpos.y * this->width + fpos.x] == 0){

          buf[fpos.y * this->width + fpos.x] = nbuf[iy * this->twidth + ix];
        }

      }
    }

    delete[] nbuf;

  }

  TIFFClose(tif);
  return true;

}

template<typename T>
bool tiff<T>::write(const char *filename) {
  
  TIFF *out= TIFFOpen(filename, "w");

  TIFFSetField(out, TIFFTAG_IMAGEWIDTH, this->width);
  TIFFSetField(out, TIFFTAG_IMAGELENGTH, this->height);
  TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8*sizeof(T));
  TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, this->width));

  T* buf = this->data;
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