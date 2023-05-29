#ifndef SIMPLEWINDEROSION_WORLD
#define SIMPLEWINDEROSION_WORLD

#include <soillib/util/pool.hpp>
#include <soillib/util/slice.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/surface.hpp>
#include <soillib/map/quadtree.hpp>

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

    cellpool.reserve(512*512);
    map.nodes.emplace_back(glm::ivec2(0), glm::ivec2(512), cellpool);
//    map.nodes.emplace_back(glm::ivec2(0, 512), glm::ivec2(512), cellpool);
//    map.nodes.emplace_back(glm::ivec2(512, 0), glm::ivec2(512), cellpool);
//    map.nodes.emplace_back(glm::ivec2(512), glm::ivec2(512), cellpool);

    // Fill the Node Array

    for(auto& node: map.nodes){

      for(auto [cell, pos]: node)
        cell.massflow = 0.0f;

      // Add Gaussian

      for(auto [cell, pos]: node){
        glm::vec2 p = glm::vec2(pos)/glm::vec2(node.dimension);
        glm::vec2 c = glm::vec2(glm::vec2(node.dimension)/glm::vec2(4, 2))/glm::vec2(node.dimension);
        float d = length(p-c);
          cell.height = exp(-d*d*node.dimension.x*0.2f);
      }

    }


    float min = 0.0f;
    float max = 0.0f;

    for(auto& node: map.nodes)
    for(auto [cell, pos]: node){
      min = (min < cell.height)?min:cell.height;
      max = (max > cell.height)?max:cell.height;
    }

    for(auto& node: map.nodes)
    for(auto [cell, pos]: node){
      cell.height = 0.5*((cell.height - min)/(max - min));
    }

  }

  static soil::pool<cell> cellpool;
  static soil::map::quadtree<cell, my_index> map;

  static float mapscale;

  //Constructor

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

soil::map::quadtree<cell, my_index> World::map;
soil::pool<cell> World::cellpool;

float World::lrate = 0.1f;
float World::mapscale = 80.0f;


















#include "wind.hpp"

void World::erode(int cycles){

  for(auto& node: map.nodes)
  for(auto [cell, pos]: node){
    cell.massflow_track = 0;
    cell.momentumx_track = 0;
    cell.momentumy_track = 0;
    cell.momentumz_track = 0;
  }

  //Do a series of iterations!
  for(auto& node: map.nodes)
  for(int i = 0; i < cycles; i++){

    //Spawn New Particle on Boundary

    glm::vec2 newpos = node.pos + glm::ivec2(rand()%node.dimension.x, rand()%node.dimension.y);
    newpos += glm::vec2(rand()%1000, rand()%1000)/1000.0f;

    Wind wind(newpos);
    while(wind.fly());

  }

  //Update Fields
  for(auto& node: map.nodes)
  for(auto [cell, pos]: node){
    cell.massflow = (1.0f-lrate)*cell.massflow + lrate*cell.massflow_track;
    cell.momentumx = (1.0f-lrate)*cell.momentumx + lrate*cell.momentumx_track;
    cell.momentumy = (1.0f-lrate)*cell.momentumy + lrate*cell.momentumy_track;
    cell.momentumz = (1.0f-lrate)*cell.momentumz + lrate*cell.momentumz_track;
  }
}

#endif
