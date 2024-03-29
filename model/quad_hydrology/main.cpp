#include <soillib/soillib.hpp>
#include <soillib/io/yaml.hpp>
#include <soillib/io/tiff.hpp>
#include <soillib/io/png.hpp>

#include "source/world.hpp"
#include <iostream>
#include <csignal>

bool quit = false;

void sighandler(int signal){
  quit = true;
}

int main( int argc, char* args[] ) {

  // Load Configuration

  soil::io::yaml config("config.yaml");
  if(!config.valid()){
    std::cout<<"failed to load yaml configuration"<<std::endl;
  }

  try {
    World::config = config.As<world_c>();
  } catch(soil::io::yaml::exception e){
    std::cout<<"failed to parse yaml configuration: "<<e.what()<<std::endl; 
    return 0;
  }

  // Initialize World

  size_t SEED = time(NULL);
  if(argc >= 2)
    SEED = std::stoi(args[1]);

  World world(SEED);
  std::cout<<"SEED: "<<world.SEED<<std::endl;

  // Run Erosion

  signal(SIGINT, &sighandler);
  while(!quit && world.erode());

  // Export Images

  soil::io::tiff discharge(world.map.max - world.map.min);
  discharge.fill([&](const glm::ivec2 pos){
    return world.discharge(pos);
  });
  discharge.write("out/discharge.tiff");

  soil::io::tiff height(world.map.max - world.map.min);
  for(auto [cell, pos]: world.map)
    height[pos-world.map.min] = cell.height;
  height.write("out/height.tiff");

/*
  soil::io::tiff vegetation(world.map.max - world.map.min);
  vegetation.fill([&](const glm::ivec2 pos){
    return world.map.get(pos)->rootdensity;
  });
  vegetation.write("out/vegetation.tiff");
  */

  soil::io::png normal(world.map.max - world.map.min);
  normal.fill([&](const glm::ivec2 pos){
    if(world.oob(pos+world.map.min))
      return glm::vec4(0);
    glm::vec3 normal = world.normal(pos+world.map.min);
    normal = glm::vec3(normal.x, -normal.z, normal.y);
    normal = 0.5f*normal + 0.5f;
    return 255.0f*glm::vec4(normal, 1.0f);
  });
  normal.write("out/normal.png");

  return 0;
}
