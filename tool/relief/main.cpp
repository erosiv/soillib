#include <soillib/soillib.hpp>

#include <soillib/util/pool.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/model/surface.hpp>
#include <soillib/io/png.hpp>
#include <soillib/io/geotiff.hpp>

#include <iostream>
#include <filesystem>
#include <limits>

// Map Structs

struct cell {
  float height;
  glm::vec3 normal;
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

// Utilization Instructions

const char* help = R""""(
Usage:
  ./main <file.tiff>

Description:

  Render a .tiff file as a relief-shaded map. Output is a .png image.

)"""";

int main(int argc, char *args[]) {

  // Parse Arguments

  if(argc < 2){
    puts(help);
    return 0;
  }

  // Load Image Data, Create Map

  const auto path = std::filesystem::path(args[1]);
  soil::io::geotiff dem(path.c_str());
  world_t world(dem.dim());

  // Fill Cell Pool w. Image

  for(auto [cell, pos]: world.map)
    cell.height = dem[pos];

  for(auto [cell, pos]: world.map)
    cell.normal = soil::surface::normal(world, pos);

  // Normalize Height Map

  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::min();
  for(auto [cell, pos]: world.map){
    if(cell.height < min) min = cell.height;
    if(cell.height > max) max = cell.height;
  }

  for(auto [cell, pos]: world.map){
    cell.height = (cell.height - min)/(max - min);
  }

  // Export a Shaded Relief Map

  soil::io::png image(dem.dim());
  image.fill([&](const glm::ivec2 pos){

    glm::vec3 normal = world.map.get(pos)->normal;
    float d = glm::dot(normal, glm::normalize(glm::vec3(1, 2, 1)));

    // Clamp
    d = 0.05f + 0.9f*d;

    // Flat-Toning
    float flattone = 0.9f;
    float weight = 1.0f-normal.y;

    float h = world.map.get(pos)->height;
    weight = weight * (1.0f - h*h);

    d = (1.0f-weight) * d + weight*flattone;

  //  glm::vec3 color = glm::mix(glm::vec3(0.57, 0.87, 0.51), glm::vec3(0.87, 0.72, 0.51), 1.0f-(1.0f-h)*(1.0f-h));

  // Arial Perspective

    glm::vec3 color = glm::vec3(1);
    return 255.0f*glm::vec4(d*color, 1.0f);
  });
  image.write("relief.png");

}
