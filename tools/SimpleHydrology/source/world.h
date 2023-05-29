#ifndef SIMPLEHYDROLOGY_WORLD
#define SIMPLEHYDROLOGY_WORLD

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/dist.hpp>

#include <soillib/map/basic.hpp>
#include <soillib/util/surface.hpp>

/*
SimpleHydrology - world.h

Defines our main storage buffers,
world updating functions for erosion
and vegetation.
*/

// Raw Interleaved Cell Data

struct cell {

  float height;
  float discharge;
  float momentumx;
  float momentumy;

  float discharge_track;
  float momentumx_track;
  float momentumy_track;

  float rootdensity;

};

using my_index = soil::index::flat;

struct World {

  const size_t SEED;
  World(size_t SEED):SEED(SEED){

    soil::dist::seed(SEED);
    map.slice = { cellpool.get(map.area), map.dimension };

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

  }

  static soil::map::basic<cell, my_index> map;
  static soil::pool<cell> cellpool;

  // Parameters

  static int mapscale;
  static float lrate;

  // Main Update Methods

  static void erode(int cycles);              // Erosion Update Step

  const static inline float height(glm::ivec2 p){
    cell* c = map.get(p);
    if(c == NULL) return 0.0f;
    return c->height;
  }

  const static inline float discharge(glm::ivec2 p){
    return erf(0.4f*map.get(p)->discharge);
  }

  const static inline glm::vec3 normal(glm::ivec2 p){
    return soil::surface::normal(map, p, glm::vec3(1, World::mapscale, 1));
  }

};

int World::mapscale = 80;
float World::lrate = 0.1f;
soil::map::basic<cell, my_index> World::map(glm::ivec2(512));
soil::pool<cell> World::cellpool(World::map.area);

#include "vegetation.h"
#include "water.h"

/*
===================================================
          HYDRAULIC EROSION FUNCTIONS
===================================================
*/
void World::erode(int cycles){

  for(auto [cell, pos]: map){
    cell.discharge_track = 0;
    cell.momentumx_track = 0;
    cell.momentumy_track = 0;
  }

  //Do a series of iterations!
  for(int i = 0; i < cycles; i++){

    //Spawn New Particle

    Drop drop(glm::vec2(map.dimension)*soil::dist::vec2());
    while(drop.move(map, drop_c) && drop.interact(map, drop_c));

  }

  //Update Fields
  for(auto [cell, pos]: map){
    cell.discharge = (1.0f-lrate)*cell.discharge + lrate*cell.discharge_track;
    cell.momentumx = (1.0f-lrate)*cell.momentumx + lrate*cell.momentumx_track;
    cell.momentumy = (1.0f-lrate)*cell.momentumy + lrate*cell.momentumy_track;
  }

}

#endif
