#include <soillib/soillib.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/util/surface.hpp>
#include <soillib/io/img/png.hpp>
#include <soillib/io/img/tiff.hpp>

#include <soillib/model/physics/cascade.hpp>

struct cell {
  float height = 0.0f;
};

int main(int argc, char *argv[]) {

  // Load the Image First

  soil::io::tiff image("height.tiff");

  // Create Cell Pool, Map

  soil::pool<cell> cellpool(image.width*image.height);
  soil::map::basic<cell> map(glm::ivec2(image.width, image.height), cellpool);

  // Fill Cell Pool w. Image

  for(auto [cell, pos]: map){
    cell.height = image[pos];
    cell.height *= 256.0f;
  }

  // Manipulate Map

  for(auto [cell, pos]: map)
    soil::phys::cascade(map, pos);

  // Export a Shaded Relief Map

  soil::io::png image_out(image.width, image.height);
  image_out.fill([&](const glm::ivec2 pos){
    glm::vec3 normal = soil::surface::normal(map, pos, glm::vec3(1, 1, 1));
    normal = glm::vec3(normal.x, -normal.z, normal.y);
    normal = 0.5f*normal + 0.5f;
    return 255.0f*glm::vec4(normal, 1.0f);
  });
  image_out.write("normal.png");

}
