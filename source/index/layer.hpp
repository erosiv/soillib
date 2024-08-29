#ifndef SOILLIB_MAP_LAYER
#define SOILLIB_MAP_LAYER

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/pool.hpp>
#include <soillib/util/slice.hpp>

/*==================================================
soillib map layer

rectangular layer map with run-length-encoded soil
segments, including iterators.

templated properties are defined on a per-cell and
per-segment basis, i.e. in area an volume.
==================================================*/

namespace soil {
namespace map {

// Configuration / Parameterization

struct layer_config {
  glm::ivec2 dimension = glm::ivec2(0);
  size_t max_depth = 0;
};

// Layer Segment, Iterator

template<typename S>
struct layer_segment_iterator;
template<typename S>
struct layer_segment: public S {

  layer_segment<S> *below = NULL;
  layer_segment<S> *above = NULL;

  layer_segment_iterator<S> begin() const noexcept { return layer_segment_iterator<S>(this); }
  layer_segment_iterator<S> end() const noexcept { return layer_segment_iterator<S>(NULL); }

  // Insertion Methods

  void insert_above(layer_segment<S> *seg) noexcept {
    seg->below = this;
    seg->above = above;
    above = seg;
  }

  void insert_below(layer_segment<S> *seg) noexcept {
    seg->above = this;
    seg->below = below;
    below = seg;
  }

  // Removal Methods

  void detach_above() noexcept {
    if (above != NULL)
      above->below = NULL;
    above = NULL;
  }

  void detach_below() noexcept {
    if (below != NULL)
      below->above = NULL;
    below = NULL;
  }
};

template<typename S>
struct layer_segment_iterator {

  typedef layer_segment<S> segment;

  segment *iter;

  layer_segment_iterator() noexcept: iter(NULL) {};
  layer_segment_iterator(const segment *iter) noexcept: iter(iter) {};

  // Base Operators

  const layer_segment_iterator<S> &operator++() noexcept {
    this->iter = this->iter->below;
    return *this;
  };

  const bool operator==(const layer_segment_iterator<S> &other) const noexcept {
    return this->iter == other.iter;
  };

  const bool operator!=(const layer_segment_iterator<S> &other) const noexcept {
    return !(*this == other);
  };

  const S operator*() const noexcept {
    return iter->segment;
  };
};

// Actual Cell Data (2D + 3D)

template<typename T, typename S>
struct layer_cell: public T {

  layer_segment<S> *top;

  layer_segment_iterator<S> begin() const noexcept { return layer_segment_iterator<S>(top); }
  layer_segment_iterator<S> end() const noexcept { return layer_segment_iterator<S>(NULL); }
};

// Layer Map

template<typename T, typename S, soil::index_t Index = soil::index::flat>
struct layer {

  typedef Index index;
  typedef layer_config config;

  typedef layer_segment<S> segment;
  typedef layer_cell<T, S> cell;

  const glm::ivec2 dimension;
  const size_t area = dimension.x * dimension.y;

  soil::slice<cell, Index> slice;
  soil::pool<segment> pool;

  layer(const glm::ivec2 dimension, const size_t poolsize)
      : dimension(dimension), slice(dimension), pool(poolsize) {}

  layer(const config config)
      : layer(config.dimension, config.dimension.x * config.dimension.y * config.max_depth) {}

  inline cell *get(const glm::ivec2 p) const noexcept {
    return slice.get(p);
  }

  const inline bool oob(const glm::ivec2 p) const noexcept {
    return slice.oob(p);
  }

  const inline glm::ivec2 bound() const noexcept {
    return dimension;
  }

  slice_iterator<cell, Index> begin() const noexcept { return slice.begin(); }
  slice_iterator<cell, Index> end() const noexcept { return slice.end(); }

  // Specialized Methods

  inline segment *top(const glm::ivec2 p) noexcept {
    return slice.get(p)->top;
  }

  inline void push(const glm::ivec2 p, S passed) noexcept {
    if (slice.oob(p))
      return;

    segment *top = slice.get(p)->top;
    segment *above = pool.get(passed);

    slice.get(p)->top = above;
    if (top != NULL)
      top->insert_above(above);
  }

  inline void pop(const glm::ivec2 p) noexcept {
    if (slice.oob(p))
      return;

    segment *top = slice.get(p)->top;
    if (top == NULL)
      return;

    slice.get(p)->top = top->below;
    top->detach_below();
    pool.unget(top);
  }
};

}; // namespace map
}; // namespace soil

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<soil::map::layer_config> {
  static soil::map::layer_config As(soil::io::yaml &node) {
    soil::map::layer_config config;
    config.dimension = node["dimension"].As<glm::ivec2>();
    config.max_depth = node["max-depth"].As<int>();
    return config;
  }
};

#endif
#endif
