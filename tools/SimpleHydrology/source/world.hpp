#ifndef SIMPLEHYDROLOGY_WORLD
#define SIMPLEHYDROLOGY_WORLD

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/dist.hpp>

#include <soillib/map/basic.hpp>
#include <soillib/model/surface.hpp>

#include <soillib/particle/water.hpp>
#include <soillib/particle/vegetation.hpp>

/*
SimpleHydrology - world.h

Defines our main storage buffers,
world updating functions for erosion
and vegetation.
*/

// Type Definitions

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

using ind_type = soil::index::flat;
using map_type = soil::map::basic<cell, ind_type>;

// World Configuration Data

struct world_c {

  int scale = 80;
  float lrate = 0.1f;

  map_type::config map_config = {
    glm::ivec2(512)
  };

  soil::WaterParticle_c water_config;

};

// Configuration Loading

#ifdef SOILLIB_IO_YAML

bool operator<<(world_c& conf, soil::io::yaml::node& node){
  try {
    conf.scale = node["scale"].As<int>();
    conf.lrate = node["lrate"].As<float>();
    conf.map_config << node["map"];
    conf.water_config << node["water"];
  } catch(soil::io::yaml::exception& e){
    return false;
  }
  return true;
}

#endif

struct World {

  const size_t SEED;
  soil::map::basic<cell, ind_type> map;
  soil::pool<cell> cellpool;

  static world_c config;

  // Parameters

  float no_basin = 0;
  float no_basin_track = 0;

  World(size_t SEED):
    SEED(SEED),
    map(config.map_config),
    cellpool(map.area)
  {

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

  // Main Update Methods

  void erode(int cycles);              // Erosion Update Step

  const inline bool oob(glm::ivec2 p){
    return map.oob(p);
  }

  const inline float height(glm::ivec2 p){
    cell* c = map.get(p);
    if(c == NULL) return 0.0f;
    return c->height;
  }

  const inline float discharge(glm::ivec2 p){
    return erf(0.4f*map.get(p)->discharge);
  }

  const inline glm::vec3 normal(glm::ivec2 p){
    return soil::surface::normal(*this, p, glm::vec3(1, World::config.scale, 1));
  }

};

world_c World::config;

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

  no_basin_track = 0;

  //Do a series of iterations!
  for(int i = 0; i < cycles; i++){

    //Spawn New Particle

    soil::WaterParticle drop(glm::vec2(map.dimension)*soil::dist::vec2());
    while(drop.move(*this, config.water_config) && drop.interact(*this, config.water_config));

    if(map.oob(drop.pos))
      no_basin_track++;

  }

  //Update Fields
  for(auto [cell, pos]: map){
    cell.discharge = (1.0f-config.lrate)*cell.discharge + config.lrate*cell.discharge_track;
    cell.momentumx = (1.0f-config.lrate)*cell.momentumx + config.lrate*cell.momentumx_track;
    cell.momentumy = (1.0f-config.lrate)*cell.momentumy + config.lrate*cell.momentumy_track;
  }

  no_basin = (1.0f-config.lrate)*no_basin + config.lrate*no_basin_track;

}

#endif
