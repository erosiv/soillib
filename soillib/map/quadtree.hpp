#ifndef SOILLIB_MAP_QUADTREE
#define SOILLIB_MAP_QUADTREE

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/slice.hpp>
#include <soillib/util/pool.hpp>

namespace soil {

template<typename T, soil::index_t Index>
struct node {

  glm::ivec2 pos = glm::ivec2(0);
  soil::slice<T, Index> slice;

  inline T* get(const glm::ivec2 p) const noexcept {
    return slice.get(p - pos);
  }

  const inline bool oob(const glm::ivec2 p) noexcept {
    return slice.oob(p - pos);
  }

};




const int mapscale = 80;

const int tilesize = 1024;
const int tilearea = tilesize*tilesize;
const glm::ivec2 tileres = glm::ivec2(tilesize);

const int mapsize = 1;
const int maparea = mapsize*mapsize;

const int size = mapsize*tilesize;
const int area = maparea*tilearea;
const glm::ivec2 res = glm::ivec2(size);




/*

template<typename T, soil::index_t Index>
struct map {

  node<T, Index> nodes[maparea];

  void init(){

    // Generate the Node Array

    for(int i = 0; i < mapsize; i++)
    for(int j = 0; j < mapsize; j++){

      int ind = i*mapsize + j;

      nodes[ind] = {
        tileres*ivec2(i, j),
        NULL,
      //  vertexpool.section(tilearea/lodarea, 0, glm::vec3(0), vertexpool.indices.size()),
        { cellpool.get(tilearea/lodarea), tileres/lodsize }
      };

//      indexnode(vertexpool, nodes[ind]);

    }

  }

  const inline bool oob(ivec2 p){
    if(p.x  < 0)  return true;
    if(p.y  < 0)  return true;
    if(p.x >= size)  return true;
    if(p.y >= size)  return true;
    return false;
  }

  inline node* get(ivec2 p){
    if(oob(p)) return NULL;
    p /= tileres;
    int ind = p.x*mapsize + p.y;
    return &nodes[ind];
  }

  inline cell* getCell(ivec2 p){
    if(oob(p)) return NULL;
    return get(p)->get(p);
  }

};

*/

}; // namespace quad

#endif
