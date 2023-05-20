#ifndef SOILLIB_IO_IMAGE_PNG
#define SOILLIB_IO_IMAGE_PNG

#include <soillib/soillib.hpp>

#include <stdio.h>
#include <png.h>
#include <functional>

/*
todo:
- the raw buffer allocation needs to be cleaner
- maybe make the data abstraction more general
  so that we can store different image formats
- this should somehow be templated by the bit-depth,
  meaning that we know what the return type is
  for a given image (e.g. 3 ints, 4 floats, etc)
*/

namespace soil {
namespace io {

struct image_base {};

// PNG Implementation

struct png: image_base {

  size_t width = 0;
  size_t height = 0;

  png_byte color_type;
  png_byte bit_depth = 8;
  png_bytep *row_data = NULL;

  // Raw Construction
  png(){};
  png(const char* filename){ read(filename); };
  png(const size_t width, const size_t height):width(width),height(height){ allocate(); }
  png(const glm::ivec2 res):width(res.x),height(res.y){ allocate(); }

  // Loading Construction

  void allocate(){
    if(row_data != NULL)
      return;
    row_data = new png_bytep[sizeof(png_bytep) * height];
    for(size_t y = 0; y < height; y++) {
      row_data[y] = new png_byte[bit_depth * width * 4];
    }
  }

  ~png(){
    if(row_data == NULL)
      return;
    for(size_t y = 0; y < height; y++)
      delete[] row_data[y];
    delete[] row_data;
  }

  // Read / Write Methods

  bool read(const char* filename);
  bool write(const char* filename);

  void fill(std::function<glm::ivec4(const glm::ivec2)> handle);

  // Subscript Operator

  const glm::ivec4 operator[](const glm::ivec2 pos){
    png_bytep row = row_data[pos.y];
    png_bytep px = &(row[pos.x * 4]);
    return glm::ivec4(px[0], px[1], px[2], px[3]);
  }

};


/*
  1. Easy construction method for image from maps
  2. Easy image use as temporary buffers?
  3. Should just be a dumpable piece of memory!!
*/

// Implementations

bool png::read(const char* filename){

    FILE *fp = fopen(filename, "rb");
    
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

    size_t _width        = png_get_image_width(png, info);
    size_t _height       = png_get_image_height(png, info);
    png_byte _color_type = png_get_color_type(png, info);
    png_byte _bit_depth  = png_get_bit_depth(png, info);

    // (Re)allocation

    if(  _width != width
      || _height != height
      || _color_type != color_type 
      || _bit_depth != bit_depth){

      width = _width;
      height = _height;
      bit_depth = _bit_depth;
      color_type = _color_type;

      if(row_data != NULL){
        delete[] row_data;
        row_data = NULL;
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

    size_t row_bytes = png_get_rowbytes(png,info);

    png_read_image(png, row_data);
    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);

    return true;

}

bool png::write(const char *filename) {
    int y;

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

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
      png,
      info,
      width, height,
      8,
      PNG_COLOR_TYPE_RGBA,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);

    if (!row_data)
      return false;

    png_write_image(png, row_data);
    png_write_end(png, NULL);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
    return true;

}

// Iterate over the cell positions and decide what to do with it...
// This should later be adopted to use map-iterators instead

void png::fill(std::function<glm::ivec4(const glm::ivec2)> handle){
  for(size_t y = 0; y < height; y++){
    png_bytep row = row_data[y];
    for(size_t x = 0; x < width; x++){
      glm::ivec4 color = (handle)(glm::ivec2(x, y));
      png_bytep px = &(row[x * 4]);
      px[0] = color[0];
      px[1] = color[1];
      px[2] = color[2];
      px[3] = color[3];
    }
  }
}

}; // end of namespace io
}; // end of namespace soil

#endif