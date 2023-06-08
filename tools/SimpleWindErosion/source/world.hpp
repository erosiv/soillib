#ifndef SIMPLEWINDEROSION_WORLD
#define SIMPLEWINDEROSION_WORLD

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/dist.hpp>

#include <soillib/map/basic.hpp>
#include <soillib/util/surface.hpp>

struct cell {

  float height;

  float massflow;
  float momentumx;
  float momentumy;
  float momentumz;

  float massflow_track;
  float momentumx_track;
  float momentumy_track;
  float momentumz_track;

};

using my_index = soil::index::flat;

class World{
public:

  const size_t SEED;
  World(size_t SEED):SEED(SEED){

    // Allocate the Nodes
    
    soil::dist::seed(SEED);
    map.slice = { cellpool.get(map.area), map.dimension };

    for(auto [cell, pos]: map){
      cell.height = 0.0f;
    }

    /*
    soil::noise::sampler sampler;
    sampler.source.SetFractalOctaves(8.0f);

    for(auto [cell, pos]: map){
      cell.height = sampler.get(glm::vec3(pos.x, pos.y, SEED%10000)/glm::vec3(512, 512, 1.0f));
    }

    // Normalize

    float min = 0.0f;
    float max = 0.0f;
    for(auto [cell, pos]: map){
      if(cell.height < min) min = cell.height;
      if(cell.height > max) max = cell.height;
    }

    for(auto [cell, pos]: map){
      cell.height = (cell.height - min)/(max - min);
    }
    */

  }

  static soil::map::basic<cell, my_index> map;
  static soil::pool<cell> cellpool;

  static float mapscale;
  static float lrate;

  static void erode(int cycles);               //Erode with N Particles

  const static float height(glm::ivec2 p){
    cell* c = map.get(p);
    if(c == NULL) return 0.0f;
    return c->height;
  }

  const static inline glm::vec3 normal(glm::ivec2 p){
    return soil::surface::normal(map, p, glm::vec3(1, mapscale, 1));
  }

};

float World::mapscale = 80.0f;
float World::lrate = 0.1f;

soil::map::basic<cell, my_index> World::map(glm::ivec2(512));
soil::pool<cell> World::cellpool(World::map.area);

// Erosion Code Implementation

#include "wind.hpp"

void World::erode(int cycles){

  for(auto [cell, pos]: map){
    cell.massflow_track = 0;
    cell.momentumx_track = 0;
    cell.momentumy_track = 0;
    cell.momentumz_track = 0;
  }

  //Do a series of iterations!
  for(int i = 0; i < cycles; i++){

    Wind wind(glm::vec2(map.dimension)*soil::dist::vec2());
    while(wind.fly());

  }

  //Update Fields
  for(auto [cell, pos]: map){
    cell.massflow = (1.0f-lrate)*cell.massflow + lrate*cell.massflow_track;
    cell.momentumx = (1.0f-lrate)*cell.momentumx + lrate*cell.momentumx_track;
    cell.momentumy = (1.0f-lrate)*cell.momentumy + lrate*cell.momentumy_track;
    cell.momentumz = (1.0f-lrate)*cell.momentumz + lrate*cell.momentumz_track;
  }
}

#endif
