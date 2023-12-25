template<typename T>
void construct(T& world, Buffer& posbuf, Buffer& normalbuf, Buffer& indexbuf){

  std::vector<int> indices;
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;

  for(auto [cell, pos]: world.map){
    positions.emplace_back(pos.x, cell.height, pos.y);
    normals.emplace_back(cell.normal);
  }

  const auto flatten = [&world](const glm::ivec2 p){
    return p.x * world.dim.y + p.y;
  };

  for(auto [cell, pos]: world.map){

    if(world.map.oob(pos + glm::ivec2(1)))
      continue;

    const int indexA = flatten(pos + glm::ivec2(0, 0));
    const int indexB = flatten(pos + glm::ivec2(1, 0));
    const int indexC = flatten(pos + glm::ivec2(0, 1));
    const int indexD = flatten(pos + glm::ivec2(1, 1));

    indices.push_back(indexB);
    indices.push_back(indexA);
    indices.push_back(indexC);

    indices.push_back(indexB);
    indices.push_back(indexC);
    indices.push_back(indexD);

  }

  posbuf.fill(positions);
  normalbuf.fill(normals);
  indexbuf.fill(indices);

}
