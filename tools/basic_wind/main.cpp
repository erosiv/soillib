#include <soillib/soillib.hpp>
#include <soillib/io/yaml.hpp>
#include <soillib/io/tiff.hpp>
#include <soillib/io/png.hpp>

#include "source/world.hpp"
#include <iostream>

int main( int argc, char* args[] ) {

  // Load Configuration

  soil::io::yaml config("config.yaml");
  if(!config.valid()){
    std::cout<<"failed to load yaml configuration"<<std::endl;
  }
  
  if(!(World::config << config.root)){
    std::cout<<"failed to parse yaml configuration"<<std::endl;
  }

  // Initialize World

  size_t SEED = time(NULL);
  if(argc >= 2)
    SEED = std::stoi(args[1]);

  World world(SEED);
  std::cout<<"SEED: "<<world.SEED<<std::endl;

  // Run Erosion

  size_t n_timesteps = 1024;
  size_t n_cycles = 512;

  while(n_timesteps > 0){

    std::cout<<n_timesteps--<<std::endl;
    world.erode(n_cycles);

  }

  soil::io::tiff height(world.map.dimension);
  height.fill([&](const glm::ivec2 pos){
    return world.map.get(pos)->height;
  });
  height.write("out/height.tiff");

  soil::io::png normal(world.map.dimension);
  normal.fill([&](const glm::ivec2 pos){
    glm::vec3 normal = world.normal(pos);
    normal = glm::vec3(normal.x, -normal.z, normal.y);
    normal = 0.5f*normal + 0.5f;
    return 255.0f*glm::vec4(normal, 1.0f);
  });
  normal.write("out/normal.png");

  return 0;
}
