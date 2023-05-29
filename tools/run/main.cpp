#include <soillib/soillib.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/surface.hpp>

#include <soillib/io/img/png.hpp>
#include <soillib/io/img/tiff.hpp>

struct cell {
  float height = 0.0f;
};

int main(int argc, char *argv[]) {

  // Create Cell Pool, Map

  soil::pool<cell> cellpool(1024*1024);
  soil::map::basic<cell> map(glm::ivec2(1024), cellpool);

  // Fill Cell Pool w. Noise

  soil::noise::sampler sampler;
  sampler.source.SetFractalOctaves(8.0f);

  for(auto [cell, pos]: map){
    cell.height = sampler.get(glm::vec3(pos.x, pos.y, 0.0f)/glm::vec3(1024, 1024, 1.0f));
  }

  // Normalize

  float min = 0.0f;
  float max = 0.0f;
  for(auto [cell, pos]: map){
    if(cell.height < min) min = cell.height;
    if(cell.height > max) max = cell.height;
  }

  for(auto [cell, pos]: map){
    cell.height = (cell.height - min)/(max - min);
  }

  // Now I need to actually run one of my erosion sims on this map...

  // Construct, Save Image

  soil::io::tiff tiff_image(1024, 1024);
  tiff_image.fill([&](const glm::ivec2 pos){
    return map.get(pos)->height;
  });
  tiff_image.write("out.tiff");

  soil::io::png png_image(1024, 1024);
  png_image.fill([&](const glm::ivec2 pos){
    int val = 255*map.get(pos)->height;
    return glm::ivec4(val, val, val, 255);
  });
  png_image.write("out.png");

}
