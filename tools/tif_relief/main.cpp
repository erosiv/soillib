#include <soillib/soillib.hpp>

#include <soillib/util/pool.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/model/surface.hpp>
#include <soillib/io/png.hpp>
#include <soillib/io/geotiff.hpp>

#include <iostream>

//#include <soillib/model/cascade.hpp>

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

typedef float value_t;
typedef soil::io::geotiff<value_t> geovalue_t;

int main(int argc, char *args[]) {

  if(argc < 2){
    std::cout<<"please specify input directory for dataset"<<std::endl;
    return 0;
  }
  std::string path = args[1];

  // Load the Image First

  geovalue_t height;
  height.meta(path.c_str());
  height.read(path.c_str());

  // Create Cell Pool, Map

  const glm::ivec2 dim = glm::ivec2(height.width, height.height);
  world_t world(dim);

  // Fill Cell Pool w. Image

  for(auto [cell, pos]: world.map){
    cell.height = height[pos];
  }

  for(auto [cell, pos]: world.map){
    cell.normal = soil::surface::normal(world, pos);
  }

  // Normalize Height Map

  float min = 0.0f;
  float max = 0.0f;
  for(auto [cell, pos]: world.map){
    if(cell.height < min) min = cell.height;
    if(cell.height > max) max = cell.height;
  }

  for(auto [cell, pos]: world.map){
    cell.height = (cell.height - min)/(max - min);
  }

  // Export a Shaded Relief Map

  soil::io::png image(height.width, height.height);
  image.fill([&](const glm::ivec2 pos){

    glm::vec3 normal = world.map.get(pos)->normal;
    float d = glm::dot(normal, glm::normalize(glm::vec3(1, 2, 1)));

    // Clamp
    if(d < 0) d = 0;
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
    //h = 1.0f-(1.0f-h)*(1.0f-h);
    //d = (1.0f-h)*0.9f + h*d;

    return 255.0f*glm::vec4(d*color, 1.0f);
  });
  image.write("relief.png");

  const auto iso_export = [&](const float scale){

    soil::io::tiff<float> isoheight(height.width, height.height);
    soil::io::png isoline(height.width, height.height);

    isoheight.fill([&](const glm::ivec2 pos){
      float h = world.map.get(pos)->height;
      return float((int)( h * scale)) / scale;
    });

    isoline.fill([&](const glm::ivec2 pos){

      if(!world.map.oob(pos + glm::ivec2(1, 1))){  
        float h00 = isoheight[pos + glm::ivec2(0, 0)];
        float h01 = isoheight[pos + glm::ivec2(0, 1)];
        float h10 = isoheight[pos + glm::ivec2(1, 0)];
        float h11 = isoheight[pos + glm::ivec2(1, 1)];
        if(h00 != h01 || h00 != h10 || h00 != h11){
          return glm::vec4(255, 255, 255, 255);
        }
      }
      return glm::vec4(0, 0, 0, 255);
    });

    isoline.write(("isoline_"+std::to_string((int)scale)+".png").c_str());

  };

 // iso_export(5.0f);
 // iso_export(10.0f);
 // iso_export(25.0f);
  iso_export(50.0f);
}