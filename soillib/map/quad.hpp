#ifndef SOILLIB_MAP_QUAD
#define SOILLIB_MAP_QUAD

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/slice.hpp>
#include <soillib/util/pool.hpp>

namespace soil {
namespace map {

/*==================================================
soillib map quad

multi-rectangular map consisting of a singular
contiguous chunk of memory. The memory is divided
into (possibly spatially disjoint) rectangular
segments, which are arranged spatially. This allows
for arbitrary map shapes and sizes.
==================================================*/

// Configuration / Parameterization

struct node_config {
  glm::ivec2 position = glm::ivec2(0);
  glm::ivec2 dimension = glm::ivec2(0);
};

struct quad_config {
  std::vector<node_config> nodes;
};

/*

// 

// Base Templated Quadtree-Node w. Iterator

template<typename T, soil::index_t Index> struct quadtree_node_iterator;
template<typename T, soil::index_t Index>
struct quadtree_node {

  typedef Index index;

  const glm::ivec2 pos;
  const glm::ivec2 dimension;
  const size_t area = dimension.x*dimension.y;

  soil::slice<T, Index> slice;

  quadtree_node(const glm::ivec2 pos, const glm::ivec2 dimension):pos(pos),dimension(dimension){}
  quadtree_node(const glm::ivec2 pos, const glm::ivec2 dimension, soil::pool<T>& pool):quadtree_node(pos, dimension){
    slice = {pool.get(area), dimension};
  }

  inline T* get(const glm::ivec2 p) noexcept {
    return slice.get(p - pos);
  }

  const inline bool oob(const glm::ivec2 p) noexcept {
    return slice.oob(p - pos);
  }

  quadtree_node_iterator<T, Index> begin() const noexcept { return quadtree_node_iterator<T, Index>(pos, dimension, slice.begin()); }
  quadtree_node_iterator<T, Index> end()   const noexcept { return quadtree_node_iterator<T, Index>(pos, dimension, slice.end()); }

};

template<typename T, soil::index_t Index>
struct quadtree_node_iterator {

  const glm::ivec2 pos;
  const glm::ivec2 dimension;

  slice_iterator<T, Index> iter = NULL;
  int ind = 0;

  quadtree_node_iterator() noexcept : iter(NULL){};
  quadtree_node_iterator( const glm::ivec2 pos, const::glm::ivec2 dimension, const slice_iterator<T, Index>& iter) noexcept : pos(pos),dimension(dimension),iter(iter){};

  // Base Operators

  const quadtree_node_iterator<T, Index>& operator++() noexcept {
    ++iter;
    ++ind;
    return *this;
  };

  const bool operator!=(const quadtree_node_iterator<T, Index> &other) const noexcept {
    return this->iter != other.iter;
  };

  const slice_t<T> operator*() noexcept {
      return *iter;
  };

};





*/


// Actual Quadtree Structure

template<typename T, soil::index_t Index = soil::index::flat>
struct quad {

  typedef Index index;
  typedef quad_config config;

  //typedef quadtree_node<T, Index> node_t;

  // Should be sorted into a quadtree!!

  //std::vector<quadtree_node<T, Index>> nodes;

  inline T* get(glm::ivec2 p){
  //  for(auto& node: nodes)
  //    if(!node.oob(p)) return node.get(p);
    return NULL;
  }

  const inline bool oob(glm::ivec2 p){
  //  for(auto& node: nodes)
  //    if(!node.oob(p)) return false;
    return true;
  }

};

}; // namespace map
}; // namespace soil

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<soil::map::node_config> {
  static soil::map::node_config As(soil::io::yaml& node){
    soil::map::node_config config;
    config.position = node["position"].As<glm::ivec2>();
    config.dimension = node["dimension"].As<glm::ivec2>();
    return config;
  }
};

template<>
struct soil::io::yaml::cast<soil::map::quad_config> {
  static soil::map::quad_config As(soil::io::yaml& node){
    soil::map::quad_config config;
    config.nodes = node["nodes"].As<std::vector<soil::map::node_config>>();
    return config;
  }
};

#endif

#endif
