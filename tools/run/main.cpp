#include <soillib/soillib.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/io/tiff.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/surface.hpp>

struct cell {
  float height = 0.0f;
};

int main(int argc, char *argv[]) {

  // Create Cell Pool, Map

  soil::pool<cell> cellpool(1024*1024);
  soil::map::basic<cell> map(glm::ivec2(1024), cellpool);

  // Fill Cell Pool w. Noise

  soil::noise::sampler sampler;
  sampler.source.SetFractalOctaves(6.0f);

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

  // Construct, Save Image

  soil::io::tiff image(1024, 1024);
  image.fill([&](const glm::ivec2 pos){
    return map.get(pos)->height;
  });
  image.write("out.tiff");

/*
  soil::io::tiff image_out(1024, 1024);
  image_out.fill([&](const glm::ivec2 pos){
    glm::vec3 normal = 255.0f*soil::surface::normal(map, pos, glm::vec3(1, 256, 1));
    normal = glm::abs(normal);
    return glm::ivec4(normal, 255);
  });

  image_out.write("out.png");
*/

}
