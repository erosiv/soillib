#ifndef SIMPLEHYDROLOGY_WORLD
#define SIMPLEHYDROLOGY_WORLD

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/dist.hpp>

#include <soillib/map/layer.hpp>
#include <soillib/model/surface.hpp>
#include <soillib/model/cascade.hpp>

#include <soillib/matrix/mixture.hpp>
#include <soillib/particle/water.hpp>
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

struct saturation {

  bool is_water = false;

  saturation operator+(const saturation rhs) {
    return *this;
  }

  saturation operator/(const float rhs) { 
    return *this;
  }

  saturation operator*(const float rhs) { 
    return *this;
  }

  // Concept Implementations

  const float maxdiff() const noexcept {
    return (is_water)?0.0f:0.7f;
  }

  const float settling() const noexcept {
    return 1.0f;
  }

};

using mat_type = saturation;

struct segment {

  float height;
  mat_type matrix;

  segment(){}
  segment(float height, mat_type matrix)
    :height(height),matrix(matrix){}

};

using ind_type = soil::index::flat;
using map_type = soil::map::layer<cell, segment, ind_type>;

// World Configuration Data

struct world_c {

  int scale = 80;
  float lrate = 0.1f;
  int maxcycles = 2048;
  float minbasin = 0.1f;
  float waterscale = 5.0f;

  map_type::config map_config;
  //mat_type::config mat_config;
  
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
    config.waterscale = node["water-scale"].As<float>();

    config.map_config = node["map"].As<map_type::config>();
  //  config.mat_config = node["matrix"].As<mat_type::config>();
    
    config.water_config = node["water"].As<soil::WaterParticle_c>();

    return config;
  
  }
};

#endif

struct World {

  const size_t SEED;
  soil::map::layer<cell, segment, ind_type> map;

  static world_c config;

  // Parameters

  float no_basin = 0;
  float no_basin_track = 0;

  World(size_t SEED):
    SEED(SEED),
    map(config.map_config)
  {

    //mat_type::conf = config.mat_config;

    soil::dist::seed(SEED);

    soil::noise::sampler sampler;
    sampler.source.SetFractalOctaves(8.0f);
    sampler.cfg.min = -2.0f;
    sampler.cfg.max = 2.0f;

    for(auto [cell, pos]: map){
      float height_s = sampler.get(glm::vec3(pos.x, pos.y, (SEED+0)%10000)/glm::vec3(512, 512, 1.0f));
      mat_type matrix;
     // matrix.weight[1] = 1;
      map.push(pos, segment(height_s, matrix));
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

  const inline float subheight(glm::ivec2 p){
    if(!map.oob(p)){
      if(map.top(p)->below == NULL)
        return World::config.scale*map.top(p)->height;
      else 
        return World::config.scale*map.top(p)->below->height;
    }
    return 0.0f;
  }

  const inline float maxremove(glm::ivec2 p, float h){
    if(map.oob(p))
      return h;
    if(map.top(p)->below == NULL)
      return h;
    return glm::min(h, map.top(p)->height - map.top(p)->below->height);
  }

  const inline float transfer(glm::ivec2 p){
    if(!map.oob(p))
      return matrix(p).is_water?0.0f:1.0f;
    return 1.0f;
  }

  const inline float gravity(glm::ivec2 p){
    if(!map.oob(p))
      return matrix(p).is_water?2.0f:2.0f;
    return 2.0f;
  }

  inline mat_type matrix(glm::ivec2 p){
    if(!map.oob(p))
      return map.top(p)->matrix;
    return mat_type();
  }

  const inline float add(glm::ivec2 p, float h, mat_type m){

    if(map.oob(p) || h == 0)
      return h;

    if(h < 0){

      if(!this->matrix(p).is_water){
        map.top(p)->height += h/World::config.scale;
        return h;
      }

      // Cap the Height Subtraction

      map.top(p)->height += h/World::config.scale;

      if(map.top(p)->height <= map.top(p)->below->height){
        map.pop(p);
        h = 0;
      }

      return h;

    }

    if(matrix(p).is_water == m.is_water){

      map.top(p)->height += h/World::config.scale;

    } else if(m.is_water) {

      map.push(p, segment(map.top(p)->height + h/World::config.scale, m));

    } else {

      map.top(p)->below->height += h/World::config.scale;
      map.top(p)->height += h/World::config.scale;

    }

    return h;

  }

  const inline glm::vec3 normal(glm::ivec2 p){
    return soil::surface::normal(*this, p);
  }

  const inline glm::vec3 subnormal(glm::ivec2 p){
    return soil::surface::subnormal(*this, p);
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


template<typename M>
concept cascade_matrix = requires(M m){
  { m.maxdiff() } -> std::same_as<float>;
  { m.settling() } -> std::same_as<float>;
};

// Cascadable Map Constraints

template<typename T, typename M>
concept cascade_t = requires(T t){
  { t.height(glm::ivec2()) } -> std::same_as<float>;
  { t.oob(glm::ivec2()) } -> std::same_as<bool>;
  { t.matrix(glm::ivec2()) } -> std::same_as<M>;
  { t.add(glm::ivec2(), float(), M()) } -> std::same_as<float>;
};


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

    soil::WaterParticle<mat_type> drop(glm::vec2(map.dimension)*soil::dist::vec2());
    drop.matrix = this->matrix(drop.pos);

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

      soil::phys::cascade<mat_type>(*this, drop.pos);

    }

     if(!drop.matrix.is_water)
      add(drop.pos, drop.sediment, drop.matrix);
    else 
      add(drop.pos, drop.volume*config.waterscale, drop.matrix);

    soil::phys::cascade<mat_type>(*this, drop.pos);

    // Attempt to Flood

    if(!map.oob(drop.pos) && soil::surface::normal(*this, drop.pos).y > 0.9999 && discharge(drop.pos) < 0.1){

      mat_type watermatrix;
      watermatrix.is_water = true;

      add(drop.pos, drop.volume*config.waterscale, watermatrix);
      soil::phys::cascade<mat_type>(*this, drop.pos);

    }

    if(map.oob(drop.pos))
      no_basin_track++;

  }

  // Lake Evaporation

  for(auto [cell, pos]: map){
    if(this->matrix(pos).is_water){
      add(pos, -0.001, this->matrix(pos));
      soil::phys::cascade<mat_type>(*this, pos);
    }
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
