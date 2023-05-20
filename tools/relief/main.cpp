#include <soillib/soillib.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/io/image.hpp>
#include <soillib/util/surface.hpp>

struct cell {
  float height = 0.0f;
};

int main(int argc, char *argv[]) {

  // Load the Image First

  soil::io::png image("in.png");

  // Create Cell Pool, Map

  soil::pool<cell> cellpool(image.width*image.height);
  soil::map::basic<cell> map(glm::ivec2(image.width, image.height), cellpool);

  // Fill Cell Pool w. Image

  for(auto [cell, pos]: map){
    cell.height = (float)image[pos].x/255.0f;
  }

  // Export a Surface Normal Map

  soil::io::png image_out(image.width, image.height);
  image_out.fill([&](const glm::ivec2 pos){
    glm::vec3 normal = 255.0f*soil::surface::normal(map, pos, glm::vec3(1, 256, 1));
    normal = glm::abs(normal);
    return glm::ivec4(normal, 255);
  });
  image_out.write("out.png");

}
