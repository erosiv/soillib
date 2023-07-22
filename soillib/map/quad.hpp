#ifndef SOILLIB_MAP_QUAD
#define SOILLIB_MAP_QUAD

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/slice.hpp>
#include <soillib/util/pool.hpp>

namespace soil {
namespace map {

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







/*
  This guy needs an iterator, which returns with the position
  of the corresponding node!
*/

// Actual Quadtree Structure

template<typename T, soil::index_t Index = soil::index::flat>
struct quadtree {

  typedef Index index;
  typedef quadtree_node<T, Index> node_t;

  // Should be sorted into a quadtree!!

  std::vector<quadtree_node<T, Index>> nodes;

  inline T* get(glm::ivec2 p){
    for(auto& node: nodes)
      if(!node.oob(p)) return node.get(p);
    return NULL;
  }

  const inline bool oob(glm::ivec2 p){
    for(auto& node: nodes)
      if(!node.oob(p)) return false;
    return true;
  }

};

}; // namespace map
}; // namespace soil

#endif
