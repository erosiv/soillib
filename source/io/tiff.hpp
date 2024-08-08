#ifndef SOILLIB_IO_TIFF
#define SOILLIB_IO_TIFF

#include <soillib/util/array.hpp>

#include <tiffio.h>
#include <iostream>

namespace soil {
namespace io {

//! tiff is a generic .tiff file interface for reading
//! and writing generic image data to and from disk.
//!
//! tiff supports multiple different bit-depths, as
//! well as the reading of tiled .tiff images.
//!
struct tiff {

  tiff(){}
  tiff(const char* filename){ read(filename); };

  bool meta(const char* filename);  //!< Load TIFF Metadata
  bool read(const char* filename);  //!< Read TIFF Raw Data
  bool write(const char* filename); //!< Write TIFF Raw Data

  uint32_t bits()   const { return this->_bits; }
  uint32_t width()  const { return this->_width; }
  uint32_t height() const { return this->_height; }

  soil::array array() const { return this->_array; }

protected:
  bool meta_loaded = false; //!< Flag: Is Meta-Data Loaded
  bool tiled_image = false; //!< Flag: Is Image Tiled

  uint32_t _width = 0;    //!< Image Width
  uint32_t _height = 0;   //!< Image Height
  uint32_t _bits = 0;     //!< Pixel Bit-Depth

  uint32_t _twidth = 0;   //!< Tile Width
  uint32_t _theight = 0;  //!< Tile Height

  soil::array _array;     //!< Underlying Data Buffer
};

//! Load TIFF Metadata
bool tiff::meta(const char* filename){

  TIFF* tif = TIFFOpen(filename, "r");

  if(!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &this->_width))
    return false;

  if(!TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &this->_height))
    return false;

  if(!TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &this->_bits))
    return false;
  
  if(TIFFGetField(tif, TIFFTAG_TILEWIDTH, &this->_twidth))
    this->tiled_image = true;

  if(TIFFGetField(tif, TIFFTAG_TILELENGTH, &this->_theight))
    this->tiled_image = true;

  TIFFClose(tif);

  this->meta_loaded = true;
  return true;

}

//! Read TIFF Raw Data
bool tiff::read(const char* filename){

  if(!meta_loaded){
    this->meta(filename);
  }

  // Note: TIFF is Column Major (See Reading Function Below)
  //  Therefore,
  
  auto shape = soil::shape_t<2>({this->height(), this->width()});
  if(this->bits() == 32){
    this->_array = soil::array_t<float>(shape);
  }
  if(this->bits() == 64){
    this->_array = soil::array_t<double>(shape);
  }

  TIFF* tif = TIFFOpen(filename, "r");

  // Load Tiled / Non-Tiled Images
  if(!this->tiled_image){

    auto data = std::visit([](auto&& args){
      return args.data();
    }, this->_array);

    uint8_t* buf = (uint8_t*)data;
    for(size_t row = 0; row < this->height(); row++){
      TIFFReadScanline(tif, buf, row);
      buf += this->width()*(this->bits() / 8);
    }

  }
  
  /*
  else {

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
  */


  TIFFClose(tif);
  return true;
}

//template<typename T>
bool tiff::write(const char *filename) {
  
  TIFF *out= TIFFOpen(filename, "w");

  TIFFSetField(out, TIFFTAG_IMAGEWIDTH, this->width());
  TIFFSetField(out, TIFFTAG_IMAGELENGTH, this->height());
  TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
  TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, this->bits());
  TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, this->width()));

  /*
  for(size_t row = 0; row < this->height; row++){
    TIFFReadScanline(tif, buf, row);
    buf += this->width*(this->bits / 8);
  }

  //T* buf = this->data;
  uint8_t* buf = (uint8_t*)this->_buf->data();
  for (uint32_t row = 0; row < this->height; row++) {
    if (TIFFWriteScanline(out, buf, row, 0) < 0)
      break;
    buf += this->width*(this->bits / 8);
    //buf += this->width;
  }
  */

  TIFFClose(out);
  return true; 

}

}; // end of namespace io
}; // end of namespace soil

#endif