#ifndef SIMPLEWINDEROSION_WORLD
#define SIMPLEWINDEROSION_WORLD

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/dist.hpp>

#include <soillib/map/basic.hpp>
#include <soillib/matrix/singular.hpp>

#include <soillib/model/surface.hpp>

#include <soillib/particle/wind.hpp>

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

using ind_type = soil::index::flat;
using map_type = soil::map::basic<cell, ind_type>;
using mat_type = soil::matrix::singular;

// World Configuration Data

struct world_c {

  int scale = 80;
  float lrate = 0.1f;

  map_type::config map_config = {
    glm::ivec2(512)
  };

  soil::WindParticle_c wind_config;

};

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<world_c> {
  static world_c As(soil::io::yaml& node){
    world_c config;
    config.scale = node["scale"].As<int>();
    config.lrate = node["lrate"].As<float>();
    config.map_config = node["map"].As<map_type::config>();
    config.wind_config = node["wind"].As<soil::WindParticle_c>();
    return config;
  }
};

#endif

class World{
public:

  const size_t SEED;
  soil::map::basic<cell, ind_type> map;

  static world_c config;

  World(size_t SEED):
    SEED(SEED),
    map(config.map_config)
  {

    soil::dist::seed(SEED);

    for(auto [cell, pos]: map){
      cell.massflow = 0.0f;
      cell.height = 0.0f;
    }

    // Add Gaussian

    for(auto [cell, pos]: map){
      glm::vec2 p = glm::vec2(pos)/glm::vec2(map.dimension);
      glm::vec2 c = glm::vec2(glm::vec2(map.dimension)/glm::vec2(4, 2))/glm::vec2(map.dimension);
      float d = length(p-c);
        cell.height = exp(-d*d*map.dimension.x*0.2f);
    }

    float min = 0.0f;
    float max = 0.0f;

    for(auto [cell, pos]: map){
      min = (min < cell.height)?min:cell.height;
      max = (max > cell.height)?max:cell.height;
    }

    for(auto [cell, pos]: map){
      cell.height = 0.5*((cell.height - min)/(max - min));
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

  bool erode(int cycles);               //Erode with N Particles

  const inline bool oob(glm::vec2 p){
    return map.oob(p);
  }

  const inline float height(glm::vec2 p){
    if(!map.oob(p))
      return config.scale*map.get(p)->height;
    return 0.0f;
  }

  inline mat_type matrix(glm::ivec2 p){
    return mat_type();
  }

  const inline glm::vec3 normal(glm::ivec2 p){
    return soil::surface::normal(*this, p);
  }

  const inline void add(glm::ivec2 p, float h, mat_type m){
    if(!map.oob(p))
      map.get(p)->height += h/World::config.scale;
  }

};

world_c World::config;

// Erosion Code Implementation

bool World::erode(int cycles){

  for(auto [cell, pos]: map){
    cell.massflow_track = 0;
    cell.momentumx_track = 0;
    cell.momentumy_track = 0;
    cell.momentumz_track = 0;
  }

  //Do a series of iterations!
  for(int i = 0; i < cycles; i++){

    soil::WindParticle<mat_type> wind(glm::vec2(map.dimension)*soil::dist::vec2());
   
    while(true){

      if(!wind.move(*this, config.wind_config))
        break;

      auto cell = map.get(wind.pos);
      if(cell != NULL){
        cell->momentumx_track += wind.speed.x;
        cell->momentumy_track += wind.speed.y;
        cell->momentumz_track += wind.speed.z;
        cell->massflow_track += wind.sediment;
      }

      if(!wind.interact(*this, config.wind_config))
        break;

    }

  }

  //Update Fields
  for(auto [cell, pos]: map){
    cell.massflow = (1.0f-config.lrate)*cell.massflow + config.lrate*cell.massflow_track;
    cell.momentumx = (1.0f-config.lrate)*cell.momentumx + config.lrate*cell.momentumx_track;
    cell.momentumy = (1.0f-config.lrate)*cell.momentumy + config.lrate*cell.momentumy_track;
    cell.momentumz = (1.0f-config.lrate)*cell.momentumz + config.lrate*cell.momentumz_track;
  }

  return true;

}

#endif
