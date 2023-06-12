#include <soillib/soillib.hpp>
#include <soillib/io/tiff.hpp>
#include <soillib/io/yaml.hpp>

#include "source/world.h"
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

  size_t n_timesteps = 0;
  size_t n_cycles = 1024;

  while((1.0f-(float)world.no_basin/(float)n_cycles) > 0.2 && n_timesteps < 8*1024){

    std::cout<<n_timesteps++<<" "<<1.0f-(float)world.no_basin/(float)n_cycles<<std::endl;

    world.erode(n_cycles); //Execute Erosion Cycles
    Vegetation::grow(world);     //Grow Trees
 
  }

  int n = 512;
  while(n-- > 0){

    std::cout<<n_timesteps++<<" "<<1.0f-(float)world.no_basin/(float)n_cycles<<std::endl;

    world.erode(n_cycles);          //Execute Erosion Cycles
    Vegetation::grow(world);    //Grow Trees
 
  }

  // Export Images

  soil::io::tiff discharge(world.map.dimension);
  discharge.fill([&](const glm::ivec2 pos){
    return world.discharge(pos);
  });
  discharge.write("out/discharge.tiff");

  soil::io::tiff height(world.map.dimension);
  height.fill([&](const glm::ivec2 pos){
    return world.height(pos); 
  });
  height.write("out/height.tiff");

  soil::io::tiff vegetation(world.map.dimension);
  vegetation.fill([&](const glm::ivec2 pos){
    return world.map.get(pos)->rootdensity;
  });
  vegetation.write("out/vegetation.tiff");

  return 0;
}
