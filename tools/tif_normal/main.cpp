#include <soillib/soillib.hpp>

#include <soillib/io/png.hpp>
#include <soillib/io/geotiff.hpp>

#include <soillib/map/basic.hpp>
#include <soillib/util/pool.hpp>

#include <soillib/model/surface.hpp>
#include <soillib/model/cascade.hpp>

#include <iostream>

// Cell-Type and World Type
//  Required Interface for Normal Computation

struct cell {
  float height = 0.0f;
};

struct world_t {

  const glm::ivec2 dim;
  soil::map::basic<cell> map;

  world_t(const glm::ivec2 dim)
    :dim(dim),map(dim){}

  const inline bool oob(glm::vec2 p){
    return map.oob(p);
  }

  const inline float height(glm::vec2 p){
    return map.get(p)->height;
  }

};

int main(int argc, char *args[]) {

  if(argc < 2){
    std::cout<<"please specify input directory for dataset"<<std::endl;
    return 0;
  }
  std::string path = args[1];

  // Load the Image First

  soil::io::geotiff<float> height(path.c_str());
  
  // Create World w. Cell Pool

  world_t world(glm::ivec2(height.width, height.height));

  // Fill Cell Pool w. Image Data

  for(auto [cell, pos]: world.map){
    cell.height = height[pos];
    cell.height *= 1.0f;
  }

  // Export a Shaded Relief Map

  soil::io::png normal(world.dim);
  normal.fill([&](const glm::ivec2 pos){
    glm::vec3 normal = soil::surface::normal(world, pos);
    normal = glm::vec3(normal.x, -normal.z, normal.y);
    normal = 0.5f*normal + 0.5f;
    return 255.0f*glm::vec4(normal, 1.0f);
  });
  normal.write("normal.png");

}
