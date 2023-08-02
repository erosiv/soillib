#ifndef SIMPLEHYDROLOGY_WORLD
#define SIMPLEHYDROLOGY_WORLD

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/dist.hpp>

#include <soillib/map/layer.hpp>
#include <soillib/model/surface.hpp>

#include "../../../soillib/particle/water_new.hpp"
//#include <soillib/particle/vegetation.hpp>

// Type Definitions

// Raw Interleaved Cell Data

struct cell {

  float discharge;
  float momentumx;
  float momentumy;

  float discharge_track;
  float momentumx_track;
  float momentumy_track;

  float rootdensity;

};

struct segment {

  float height;
  float type;

};

using ind_type = soil::index::flat;
using map_type = soil::map::layer<cell, segment, ind_type>;

// World Configuration Data

struct world_c {

  int scale = 80;
  float lrate = 0.1f;
  int maxcycles = 2048;
  float minbasin = 0.1f;

  map_type::config map_config;
  soil::WaterParticle_c water_config;

};

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<world_c> {
  static world_c As(soil::io::yaml& node){

    world_c config;
    
    config.scale = node["scale"].As<int>();
    config.lrate = node["lrate"].As<float>();
    config.maxcycles = node["max-cycles"].As<int>();
    config.minbasin = node["min-basin"].As<float>();

    config.map_config = node["map"].As<map_type::config>();
    config.water_config = node["water"].As<soil::WaterParticle_c>();
    
    return config;
  
  }
};

#endif

struct World {

  const size_t SEED;
  soil::map::layer<cell, segment, ind_type> map;
  soil::pool<soil::map::layer_cell<cell, segment>> cellpool;
  soil::pool_t<soil::map::layer_segment<segment>> segpool;

  static world_c config;

  // Parameters

  float no_basin = 0;
  float no_basin_track = 0;

  World(size_t SEED):
    SEED(SEED),
    map(config.map_config),
    cellpool(map.area),
    segpool(map.area*16)
  {

    soil::dist::seed(SEED);

    map.slice = { cellpool.get(map.area), map.dimension };

    soil::noise::sampler sampler;
    sampler.source.SetFractalOctaves(8.0f);
    sampler.cfg.min = -2.0f;
    sampler.cfg.max = 2.0f;

    for(auto [cell, pos]: map){
      float height_s = sampler.get(glm::vec3(pos.x, pos.y, (SEED+0)%10000)/glm::vec3(512, 512, 1.0f));
      float type_s = sampler.get(glm::vec3(pos.x, pos.y, (SEED+1)%10000)/glm::vec3(512, 512, 1.0f));
      segment seg = {
        height_s,
        0.5f + 0.5f*type_s
      };
      cell.top = segpool.get(seg);
    }

    // Normalize

    float min = 0.0f;
    float max = 0.0f;
    for(auto [cell, pos]: map){
      if(cell.top->height < min) min = cell.top->height;
      if(cell.top->height > max) max = cell.top->height;
    }

    for(auto [cell, pos]: map){
      cell.top->height = (cell.top->height - min)/(max - min);
      if(cell.top->type > 1) cell.top->type = 1;
      if(cell.top->type < 0) cell.top->type = 0;
    }

  }

  // Main Update Methods

  bool erode();

  // Interface Methods

  const inline bool oob(glm::ivec2 p){
    return map.oob(p);
  }

  const inline float height(glm::ivec2 p){
    if(!map.oob(p))
      return World::config.scale*map.top(p)->height;
    return 0.0f;
  }

  const inline float type(glm::ivec2 p){
    if(!map.oob(p))
      return map.top(p)->type;
    return 0.0f;
  }

  const inline void add(glm::ivec2 p, float h, float t){

    if(map.oob(p))
      return;

    const float mrate = 0.0025f;

    if(h > 0){
      float s = h/World::config.scale + mrate;
      map.top(p)->type = (h/World::config.scale*t + mrate*map.top(p)->type)/s;
    }

    map.top(p)->height += h/World::config.scale;

  }

  const inline glm::vec3 normal(glm::ivec2 p){
    return soil::surface::normal(*this, p);
  }

  const inline glm::vec2 momentum(glm::ivec2 p){
    if(!map.oob(p)){
      return glm::vec2(map.get(p)->momentumx, map.get(p)->momentumy);
    }
    return glm::vec2(0);
  }

  const inline float discharge(glm::ivec2 p){
    if(!map.oob(p))
      return erf(0.4f*map.get(p)->discharge);
    return 0.0f;
  }

  const inline float resistance(glm::ivec2 p){
    if(!map.oob(p))
      return map.get(p)->rootdensity;
    return 0.0f;
  }

};

world_c World::config;

/*
===================================================
          HYDRAULIC EROSION FUNCTIONS
===================================================
*/

int n_timesteps = 0;

bool World::erode(){

  const float ncycles = map.area/World::config.water_config.maxAge;

  std::cout<<n_timesteps++<<" "<<1.0f-(float)no_basin/(float)ncycles<<std::endl;

  if(n_timesteps > config.maxcycles)
    return false;

  if((1.0f-(float)no_basin/(float)ncycles) < config.minbasin)
    return false;

  for(auto [cell, pos]: map){
    cell.discharge_track = 0;
    cell.momentumx_track = 0;
    cell.momentumy_track = 0;
  }

  no_basin_track = 0;

  // Max-Age is Mean-Path-Length,

  //Do a series of iterations!
  for(int i = 0; i < ncycles; i++){

    //Spawn New Particle

    soil::WaterParticle drop(glm::vec2(map.dimension)*soil::dist::vec2());
    drop.type = this->type(drop.pos);

    while(true){

      if(!drop.move(*this, config.water_config))
        break;

      // Update Discharge, Momentum Tracking Maps

      auto cell = map.get(drop.pos);
      if(cell != NULL){
        cell->discharge_track += drop.volume;
        cell->momentumx_track += drop.volume*drop.speed.x;
        cell->momentumy_track += drop.volume*drop.speed.y;
      }

      if(!drop.interact(*this, config.water_config))
        break;

    }

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
  //Vegetation::grow(*this);     //Grow Trees

  return true;

}

#endif
