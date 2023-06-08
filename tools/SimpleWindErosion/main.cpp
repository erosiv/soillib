#include <soillib/soillib.hpp>
#include <soillib/io/img/tiff.hpp>

#include "source/world.hpp"
#include <iostream>

int main( int argc, char* args[] ) {

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

  soil::io::tiff tiff_image(512, 512);
  tiff_image.fill([&](const glm::ivec2 pos){
    return world.map.get(pos)->height;
  });
  tiff_image.write("out.tiff");

  return 0;
}
