#ifndef SOILLIB_PARTICLE_VEGETATION
#define SOILLIB_PARTICLE_VEGETATION

#include <soillib/util/dist.hpp>

struct Plant {

  Plant(glm::vec2 _pos) { pos = _pos; };

  // Properties

  glm::vec2 pos;
  float size = 0.0;

  // Parameters

  static float maxSize;
  static float growRate;
  static float maxSteep;
  static float maxDischarge;
  static float maxTreeHeight;

  // Update Functions

  template<typename T>
  void root(T &world, float factor);
  void grow();

  template<typename T>
  static bool spawn(T &world, glm::vec2 pos);

  template<typename T>
  bool die(T &world);
};

float Plant::maxSize = 1.5f;
float Plant::growRate = 0.05f;
float Plant::maxSteep = 0.8f;
float Plant::maxDischarge = 0.3f;
float Plant::maxTreeHeight = 100.0f;

// Vegetation Struct (Plant Container)

struct Vegetation {

  static std::vector<Plant> plants;

  template<typename T>
  static bool grow(T &world);
};

std::vector<Plant> Vegetation::plants;

/*
================================================================================
                      Vegetation Method Implementations
================================================================================
*/

// Plant Specific Methods

void Plant::grow() {
  size += growRate * (maxSize - size);
};

template<typename T>
bool Plant::die(T &world) {

  if (world.discharge(pos) >= Plant::maxDischarge)
    return true;
  if (world.height(pos) >= Plant::maxTreeHeight)
    return true;
  if (soil::dist::istrue(1E-4))
    return true;
  return false;
}

template<typename T>
bool Plant::spawn(T &world, glm::vec2 pos) {

  if (world.discharge(pos) >= Plant::maxDischarge)
    return false;
  glm::vec3 n = world.normal(pos);
  if (n.y < Plant::maxSteep)
    return false;
  if (world.height(pos) >= Plant::maxTreeHeight)
    return false;

  return true;
}

template<typename T>
void Plant::root(T &world, float f) {

  auto c = world.map.get(pos + glm::vec2(0, 0));
  if (c != NULL)
    c->rootdensity += f * 1.0f;

  c = world.map.get(pos + glm::vec2(1, 0));
  if (c != NULL)
    c->rootdensity += f * 0.6f;

  c = world.map.get(pos + glm::vec2(-1, 0));
  if (c != NULL)
    c->rootdensity += f * 0.6f;

  c = world.map.get(pos + glm::vec2(0, 1));
  if (c != NULL)
    c->rootdensity += f * 0.6f;

  c = world.map.get(pos + glm::vec2(0, -1));
  if (c != NULL)
    c->rootdensity += f * 0.6f;

  c = world.map.get(pos + glm::vec2(-1, -1));
  if (c != NULL)
    c->rootdensity += f * 0.4f;

  c = world.map.get(pos + glm::vec2(1, -1));
  if (c != NULL)
    c->rootdensity += f * 0.4f;

  c = world.map.get(pos + glm::vec2(-1, 1));
  if (c != NULL)
    c->rootdensity += f * 0.4f;

  c = world.map.get(pos + glm::vec2(1, 1));
  if (c != NULL)
    c->rootdensity += f * 0.4f;
}

// Vegetation Specific Methods

template<typename T>
bool Vegetation::grow(T &world) {

  for (size_t i = 0; i < 5; i++) {

    glm::vec2 p = glm::vec2(world.map.dimension) * soil::dist::vec2();
    if (Plant::spawn(world, p)) {

      plants.emplace_back(p);
      plants.back().root(world, 1.0);
    }
  }

  // Iterate over Plants

  for (int i = 0; i < plants.size(); i++) {

    // Grow the Plant

    plants[i].grow();

    // Check for Kill Plant

    if (plants[i].die(world)) {

      plants[i].root(world, -1.0);
      plants.erase(plants.begin() + i);
      i--;
      continue;
    }

    // Check for Growth

    if (soil::dist::istrue(0.99))
      continue;

    // Find New Position
    glm::vec2 npos = plants[i].pos - glm::vec2(4) + glm::vec2(9) * soil::dist::vec2();

    // Check for Out-Of-Bounds
    if (world.map.oob(npos))
      continue;

    if (world.discharge(npos) >= Plant::maxDischarge)
      continue;

    if (soil::dist::istrue(world.map.get(npos)->rootdensity))
      continue;

    glm::vec3 n = world.normal(npos);

    if (n.y <= Plant::maxSteep)
      continue;

    plants.emplace_back(npos);
    plants.back().root(world, 1.0);
  }

  return true;
};

#endif
