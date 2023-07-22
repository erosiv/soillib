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

// Configuration Loading

#ifdef SOILLIB_IO_YAML

bool operator<<(node_config& conf, soil::io::yaml::node& node){
  try {
    conf.position.x = node["position"][0].As<int>();
    conf.position.y = node["position"][1].As<int>();
    conf.dimension.x = node["dimension"][0].As<int>();
    conf.dimension.y = node["dimension"][1].As<int>();
  } catch(soil::io::yaml::exception& e){
    return false;
  }
  return true;
}

bool operator<<(quad_config& conf, soil::io::yaml::node& node){
  try {
    auto nodes = node["nodes"];
    for(const auto& yaml_node: nodes){
      node_config node;
      node << yaml_node;
      quad_config.nodes.push_back(node);
    }
  } catch(soil::io::yaml::exception& e){
    return false;
  }
  return true;
}

#endif

}; // namespace map
}; // namespace soil

#endif
