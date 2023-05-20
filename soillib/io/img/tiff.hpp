#ifndef SOILLIB_IO_IMAGE_TIFF
#define SOILLIB_IO_IMAGE_TIFF

#include <soillib/io/img/img.hpp>
#include <tiffio.h>

namespace soil {
namespace io {

// TIFF Implementation

template<typename T = float>
struct tiff: soil::io::img<T> {

  using soil::io::img<T>::allocate;
  using soil::io::img<T>::operator[];

  tiff(){};
  tiff(const char* filename){ read(filename); };
  tiff(const size_t width, const size_t height):img<T>(width, height){ allocate(); }
  tiff(const glm::ivec2 res):img<T>(res){ allocate(); }

  bool read(const char* filename);
  bool write(const char* filename);

};

// Implementations

template<typename T>
bool tiff<T>::read(const char* filename){
    return true;
}

template<typename T>
bool tiff<T>::write(const char *filename) {
  
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