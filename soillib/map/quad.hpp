#ifndef SOILLIB_MAP_QUAD
#define SOILLIB_MAP_QUAD

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/slice.hpp>

#include <vector>
#include <climits>

/*==================================================
soillib map quad

multi-rectangular map consisting of a singular
contiguous chunk of memory. The memory is divided
into (possibly spatially disjoint) rectangular
segments, which are arranged spatially. This allows
for arbitrary map shapes and sizes.
==================================================*/

namespace soil {
namespace map {

// Configuration / Parameterization

struct quad_node_config {
  glm::ivec2 position = glm::ivec2(0);
  glm::ivec2 dimension = glm::ivec2(0);
};

struct quad_config {
  std::vector<quad_node_config> nodes;
};

// Base Templated Quad-Node w. Iterator

template<typename T, soil::index_t Index>
struct quad_node {

  typedef Index index;
  typedef quad_node_config config;

  const glm::ivec2 position;
  const glm::ivec2 dimension;
  const size_t area = dimension.x*dimension.y;

  soil::slice<T, Index> slice;

  quad_node(const glm::ivec2 position, const glm::ivec2 dimension)
    :position(position),dimension(dimension),slice(dimension){}

  quad_node(const config config)
    :quad_node(config.position,config.dimension){}

  inline T* get(const glm::ivec2 p) noexcept {
    return slice.get(p - position);
  }

  const inline bool oob(const glm::ivec2 p) noexcept {
    return slice.oob(p - position);
  }

  const inline glm::ivec2 min() const noexcept {
    return position;
  }

  const inline glm::ivec2 max() const noexcept {
    return position + dimension;
  }

  slice_iterator<T, Index> begin() const noexcept { return slice.begin(); }
  slice_iterator<T, Index> end()   const noexcept { return slice.end(); }

};

// Actual Quad Map Structure w. Iterator

template<typename T, soil::index_t Index> struct quad_iterator;
template<typename T, soil::index_t Index = soil::index::flat>
struct quad {

  typedef Index index;
  typedef quad_config config;
  typedef quad_node<T, Index> node;

  size_t area = 0;
  glm::ivec2 min = glm::ivec2(INT_MAX);
  glm::ivec2 max = glm::ivec2(INT_MIN);

  std::vector<node*> nodes;

  quad(const config config){

    // Initialze Min, Max Extents

    // Initialize Node-Set, Compute Area

    for(auto& node_config: config.nodes){
      nodes.push_back(new node(node_config));
      area += nodes.back()->area;
      min = glm::min(min, nodes.back()->min());
      max = glm::max(max, nodes.back()->max());
    }

  }

  ~quad(){
    for(auto& node: nodes)
    if(node != NULL){
      delete node;
      node = NULL;
    }
  }

  // Note: This access pattern is currently slow!

  inline T* get(glm::ivec2 p){
    for(auto& node: nodes)
      if(!node->oob(p)) return node->get(p);
    return NULL;
  }

  const inline bool oob(glm::ivec2 p){
    for(auto& node: nodes)
      if(!node->oob(p)) return false;
    return true;
  }

  quad_iterator<T, Index> begin() const noexcept { return quad_iterator<T, Index>(nodes.begin()); }
  quad_iterator<T, Index> end()   const noexcept { return quad_iterator<T, Index>(nodes.end()); }

};

template<typename T, soil::index_t Index>
struct quad_iterator {

  typedef quad_node<T, Index> node;
  
  std::vector<node*>::const_iterator iter = NULL;
  slice_iterator<T, Index> slice_iter;

  quad_iterator() noexcept : iter(NULL){};
  quad_iterator(const std::vector<node*>::const_iterator& iter) noexcept : 
    iter(iter),slice_iter((*iter)->begin()){};

  // Base Operators

  const quad_iterator<T, Index>& operator++() noexcept {
    if(++slice_iter == (*iter)->end()){
      ++iter;
      slice_iter = (*iter)->begin();
    }
    return *this;
  };

  const bool operator==(const quad_iterator<T, Index> &other) const noexcept {
    if(iter != other.iter) return false;
    if(slice_iter != other.slice_iter) return false;
    return true;
  };

  const bool operator!=(const quad_iterator<T, Index> &other) const noexcept {
    return !(*this == other);
  };

  const slice_t<T> operator*() noexcept {
    slice_t<T> deref = *slice_iter;
    return {deref.start, deref.pos + (*iter)->position};
  };

};

}; // namespace map
}; // namespace soil

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<soil::map::quad_node_config> {
  static soil::map::quad_node_config As(soil::io::yaml& node){
    soil::map::quad_node_config config;
    config.position = node["position"].As<glm::ivec2>();
    config.dimension = node["dimension"].As<glm::ivec2>();
    return config;
  }
};

template<>
struct soil::io::yaml::cast<soil::map::quad_config> {
  static soil::map::quad_config As(soil::io::yaml& node){
    soil::map::quad_config config;
    config.nodes = node["nodes"].As<std::vector<soil::map::quad_node_config>>();
    return config;
  }
};

#endif

#endif
