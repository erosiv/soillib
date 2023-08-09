#include <soillib/soillib.hpp>

#include <soillib/util/pool.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/model/surface.hpp>
#include <soillib/io/png.hpp>
#include <soillib/io/tiff.hpp>

#include <iostream>

//#include <soillib/model/cascade.hpp>

struct cell {
  float height = 0.0f;
  glm::vec3 normal;
};

int main(int argc, char *args[]) {

  if(argc < 2){
    std::cout<<"please specify input directory for dataset"<<std::endl;
    return 0;
  }
  std::string path = args[1];

  // Load the Image First

  soil::io::tiff height((path + "/height.tiff").c_str());
  soil::io::png normal((path + "/normal.png").c_str());

  // Create Cell Pool, Map

  const glm::ivec2 dim = glm::ivec2(height.width, height.height);
  soil::map::basic<cell> map(dim);

  // Fill Cell Pool w. Image

  for(auto [cell, pos]: map){
    cell.height = height[pos];
    cell.normal = glm::normalize(2.0f*glm::vec3(normal[pos])/255.0f - 1.0f);
    cell.normal = glm::vec3(cell.normal.x, cell.normal.z, cell.normal.y);
  }

  // Normalize Height Map

  float min = 0.0f;
  float max = 0.0f;
  for(auto [cell, pos]: map){
    if(cell.height < min) min = cell.height;
    if(cell.height > max) max = cell.height;
  }

  for(auto [cell, pos]: map){
    cell.height = (cell.height - min)/(max - min);
  }

  // Export a Shaded Relief Map

  soil::io::png image(height.width, height.height);
  image.fill([&](const glm::ivec2 pos){

    glm::vec3 normal = map.get(pos)->normal;
    float d = glm::dot(normal, glm::normalize(glm::vec3(-1, 2, -1)));

    // Clamp
    d = 0.05f + 0.9f*d;

    // Flat-Toning
    float flattone = 0.9f;
    float weight = 1.0f-normal.y;

    float h = map.get(pos)->height;
    weight = weight * (1.0f - h*h);

    d = (1.0f-weight) * d + weight*flattone;

  //  glm::vec3 color = glm::mix(glm::vec3(0.57, 0.87, 0.51), glm::vec3(0.87, 0.72, 0.51), 1.0f-(1.0f-h)*(1.0f-h));

  // Arial Perspective

    glm::vec3 color = glm::vec3(1);
    h = 1.0f-(1.0f-h)*(1.0f-h);
    d = (1.0f-h)*0.9f + h*d;

    return 255.0f*glm::vec4(d*color, 1.0f);
  });
  image.write("relief.png");

}
