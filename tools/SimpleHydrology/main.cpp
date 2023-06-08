#include <soillib/soillib.hpp>
#include <soillib/io/img/tiff.hpp>

#include "source/world.h"
#include <iostream>

int main( int argc, char* args[] ) {

  // Initialize World

  size_t SEED = time(NULL);
  if(argc >= 2)
    SEED = std::stoi(args[1]);

  World world(SEED);
  std::cout<<"SEED: "<<world.SEED<<std::endl;

  // Run Erosion

  size_t n_timesteps = 0;
  size_t n_cycles = 1024;

  while((1.0f-(float)World::no_basin/(float)n_cycles) > 0.4 && n_timesteps < 8*1024){

    std::cout<<n_timesteps++<<" "<<1.0f-(float)World::no_basin/(float)n_cycles<<std::endl;

    world.erode(n_cycles); //Execute Erosion Cycles
    Vegetation::grow();     //Grow Trees
 
  }

  int n = 512;
  while(n-- > 0){

    std::cout<<n_timesteps++<<" "<<1.0f-(float)World::no_basin/(float)n_cycles<<std::endl;

    world.erode(n_cycles); //Execute Erosion Cycles
    Vegetation::grow();     //Grow Trees
 
  }

  // Export Images

  soil::io::tiff discharge(1024, 1024);
  discharge.fill([&](const glm::ivec2 pos){
    return erf(0.4f*world.map.get(pos)->discharge);
  });
  discharge.write("discharge.tiff");

  soil::io::tiff height(1024, 1024);
  height.fill([&](const glm::ivec2 pos){
    return world.map.get(pos)->height;
  });
  height.write("height.tiff");

  soil::io::tiff vegetation(1024, 1024);
  vegetation.fill([&](const glm::ivec2 pos){
    return world.map.get(pos)->rootdensity;
  });
  vegetation.write("vegetation.tiff");

  return 0;
}
