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

  soil::io::tiff image("in.tiff");

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
    float d = glm::dot(normal, glm::normalize(glm::vec3(-1, 2, -1)));

    // Clamp
    d = 0.05f + 0.9f*d;

    // Flat-Toning
    float flattone = 0.9f;
    float weight = 1.0f-normal.y;

    float h = map.get(pos)->height/256.0f;
    weight = weight * (1.0f - h*h);

    d = (1.0f-weight) * d + weight*flattone;

  //  glm::vec3 color = glm::mix(glm::vec3(0.57, 0.87, 0.51), glm::vec3(0.87, 0.72, 0.51), 1.0f-(1.0f-h)*(1.0f-h));

  // Arial Perspective

    glm::vec3 color = glm::vec3(1);
    h = 1.0f-(1.0f-h)*(1.0f-h);
    d = (1.0f-h)*0.9f + h*d;

    return 255.0f*glm::vec4(d*color, 1.0f);
  });
  image_out.write("out.png");

}
