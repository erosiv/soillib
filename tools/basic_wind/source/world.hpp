#ifndef SIMPLEWINDEROSION_WORLD
#define SIMPLEWINDEROSION_WORLD

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/noise.hpp>
#include <soillib/util/dist.hpp>

#include <soillib/map/basic.hpp>
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

bool operator<<(world_c& conf, soil::io::yaml::node& node){
  try {
    conf.scale = node["scale"].As<int>();
    conf.lrate = node["lrate"].As<float>();
    conf.map_config << node["map"];
    conf.wind_config << node["wind"];
  } catch(soil::io::yaml::exception& e){
    return false;
  }
  return true;
}

#endif

class World{
public:

  const size_t SEED;
  soil::map::basic<cell, ind_type> map;
  soil::pool<cell> cellpool;

  static world_c config;

  World(size_t SEED):
    SEED(SEED),
    map(config.map_config),
    cellpool(map.area)
  {

    soil::dist::seed(SEED);
    map.slice = { cellpool.get(map.area), map.dimension };

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

  void erode(int cycles);               //Erode with N Particles

  const inline bool oob(glm::ivec2 p){
    return map.oob(p);
  }

  const float height(glm::ivec2 p){
    cell* c = map.get(p);
    if(c == NULL) return 0.0f;
    return c->height;
  }

  const inline glm::vec3 normal(glm::ivec2 p){
    return soil::surface::normal(*this, p, glm::vec3(1, World::config.scale, 1));
  }

};

world_c World::config;

// Erosion Code Implementation

void World::erode(int cycles){

  for(auto [cell, pos]: map){
    cell.massflow_track = 0;
    cell.momentumx_track = 0;
    cell.momentumy_track = 0;
    cell.momentumz_track = 0;
  }

  //Do a series of iterations!
  for(int i = 0; i < cycles; i++){

    soil::WindParticle wind(glm::vec2(map.dimension)*soil::dist::vec2());
    while(wind.move(*this, soil::wind_c) && wind.interact(*this, soil::wind_c));

  }

  //Update Fields
  for(auto [cell, pos]: map){
    cell.massflow = (1.0f-config.lrate)*cell.massflow + config.lrate*cell.massflow_track;
    cell.momentumx = (1.0f-config.lrate)*cell.momentumx + config.lrate*cell.momentumx_track;
    cell.momentumy = (1.0f-config.lrate)*cell.momentumy + config.lrate*cell.momentumy_track;
    cell.momentumz = (1.0f-config.lrate)*cell.momentumz + config.lrate*cell.momentumz_track;
  }
}

#endif
