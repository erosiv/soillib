#ifndef SOILLIB_IO_IMAGE_TIFF
#define SOILLIB_IO_IMAGE_TIFF

#include <soillib/soillib.hpp>
#include <soillib/io/png.hpp>
#include <soillib/util/dist.hpp>
#include <functional>
#include <tiffio.h>

namespace soil {
namespace io {

// TIFF Implementation

struct tiff: image_base {

  size_t width = 0;
  size_t height = 0;

  float* data = NULL;

  // Raw Construction
  tiff(){};
  tiff(const char* filename){ read(filename); };
  tiff(const size_t width, const size_t height):width(width),height(height){ allocate(); }
  tiff(const glm::ivec2 res):width(res.x),height(res.y){ allocate(); }

  // Loading Construction

  void allocate(){
    if(data != NULL)
      return;
    data = new float[width * height];
  }

  ~tiff(){
    if(data == NULL)
      return;
    delete[] data;
  }

  // Read / Write Methods

  bool read(const char* filename);
  bool write(const char* filename);

  void fill(std::function<float(const glm::ivec2)> handle);

  // Subscript Operator

  const glm::ivec4 operator[](const glm::ivec2 pos){
    float* px = &data[pos.y*width + pos.x];
    return glm::ivec4(px[0], px[1], px[2], px[3]);
  }

};


/*
  1. Easy construction method for image from maps
  2. Easy image use as temporary buffers?
  3. Should just be a dumpable piece of memory!!
*/

// Implementations

bool tiff::read(const char* filename){
    return true;
}

bool tiff::write(const char *filename) {
  
  TIFF *out= TIFFOpen(filename, "w");
  TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);   // set number of channels per pixel
  TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);    // set the size of the channels
  TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
  //   Some other essential fields to set that you do not have to understand for now.
  TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
  // We set the strip size of the file to be size of one row of pixels
  TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width));

  float* buf = data;
  //Now writing image to the file one strip at a time
  for (uint32 row = 0; row < height; row++) {
      if (TIFFWriteScanline(out, buf, row, 0) < 0)
      break;
    buf += width;
  }

  TIFFClose(out);
  return true; 
}

// Iterate over the cell positions and decide what to do with it...
// This should later be adopted to use map-iterators instead

void tiff::fill(std::function<float(const glm::ivec2)> handle){
  for(size_t y = 0; y < height; y++){
    for(size_t x = 0; x < width; x++){
      data[y*width + x] = (handle)(glm::ivec2(x,y));
      //data[y*width + x] = 0xffffffff;//soil::dist::uniform();//(handle)(glm::ivec2(x,y));
    }
  }
}

}; // end of namespace io
}; // end of namespace soil

#endif