#ifndef SOILLIB_IO_PNG
#define SOILLIB_IO_PNG

#include <soillib/io/img.hpp>
#include <stdio.h>
#include <png.h>

/*
todo:
- make some default color space typedefs
- improve reallocation / bad allocation handling
    e.g. tvec4<uint8_t> trying to read 24 bit depth image
*/

// PNG Implementation

namespace soil {
namespace io {

template<typename T = glm::tvec4<uint8_t>>
struct png: soil::io::img<T> {

  using soil::io::img<T>::allocate;
  using soil::io::img<T>::operator[];

  png(){};
  png(const char* filename){ read(filename); };
  png(const size_t width, const size_t height):img<T>(width, height){ allocate(); }
  png(const glm::ivec2 res):img<T>(res){ allocate(); }

  bool read(const char* filename);
  bool write(const char* filename);

};

// Read / Write Implementations

template<typename T>
bool png<T>::read(const char* filename){

    FILE *fp = fopen(filename, "rb");
    if(!fp)
      return false;
    
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) 
      return false;

    png_infop info = png_create_info_struct(png);
    if(!info) 
      return false;

    if(setjmp(png_jmpbuf(png))) 
      return false;

    png_init_io(png, fp);
    png_read_info(png, info);

    // Note: This aspect where allocation occurs on demand needs to be reconsidered.

    size_t _width       = png_get_image_width(png, info);
    size_t _height      = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    // (Re)allocation

    if(  _width != this->width
      || _height != this->height){
    //  || _color_type != color_type 
    //  || _bit_depth != bit_depth){

      this->width = _width;
      this->height = _height;
      //bit_depth = _bit_depth;
      //color_type = _color_type;

      if(this->data != NULL){
        delete[] this->data;
        this->data = NULL;
      }

      allocate();
    
    }

    if(bit_depth == 16)
      png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
      png_set_palette_to_rgb(png);

    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
      png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
      png_set_tRNS_to_alpha(png);

    if( color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
      png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if( color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
      png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    // Read Raw Data

    size_t row_bytes = png_get_rowbytes(png,info);

    png_bytep* row_pointers = new png_bytep[this->height];
    for(size_t h = 0; h < this->height; h++)
      row_pointers[h] = (png_bytep)&(this->data[h*this->width]);

    png_read_image(png, row_pointers);

    // Clean-Up

    delete[] row_pointers;
    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);

    return true;

}

template<typename T>
bool png<T>::write(const char *filename) {

    FILE *fp = fopen(filename, "wb");
    if(!fp)
      return false;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png)
      return false;

    png_infop info = png_create_info_struct(png);
    if (!info)
      return false;

    if (setjmp(png_jmpbuf(png)))
      return false;

    png_init_io(png, fp);

    png_set_IHDR(
      png,
      info,
      this->width, this->height,
      8,
      PNG_COLOR_TYPE_RGBA,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // Write Raw Data

    if (!this->data)
      return false;

    png_bytep* row_pointers = new png_bytep[this->height];
    for(size_t h = 0; h < this->height; h++)
      row_pointers[h] = (png_bytep)&(this->data[h*this->width]);

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    // Clean-Up

    delete[] row_pointers;
    fclose(fp);
    png_destroy_write_struct(&png, &info);

    return true;

}

}; // end of namespace io
}; // end of namespace soil

#endif