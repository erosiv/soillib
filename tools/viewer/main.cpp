#include <TinyEngine/TinyEngine>
#include <TinyEngine/color>
#include <TinyEngine/camera>
#include <TinyEngine/image>

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/io/geotiff.hpp>

#include <soillib/model/surface.hpp>

#include "model.hpp"
#include <limits>
#include <filesystem>

// Map Structs

struct cell {
  float height;
  glm::vec3 normal;
};

struct world_t {

  const glm::ivec2 dim;
  soil::map::basic<cell> map;

  world_t(const glm::ivec2 dim)
    :dim(dim),map(dim){}

  const inline bool oob(glm::vec2 p){
    return map.oob(p);
  }

  const inline float height(glm::vec2 p){
    return map.get(p)->height;
  }

};

// Image Structs

typedef float value_t;
typedef soil::io::geotiff<value_t> geotiff_t;

struct image_t {
  std::string name;
  geotiff_t* tiff;
};

// Utilization Instructions

const char* help = R""""(
Usage:
  ./main <file.tiff|dir>

Description:
  Render a .tiff file as a height-map, or render a directory of .tiff files
  as a merged height-map. Note that in order to render a directory of .tiff
  files, they must have GeoTIFF tags for computing their relative position.
)"""";

int main(int argc, char* args[]){

  // Parse Arguments

  if(argc < 2){
    puts(help);
    return 0;
  }

  // Load Image Data, Create Map

  std::vector<image_t> images;

  const auto path = std::filesystem::path(args[1]);
  if(!std::filesystem::is_directory(path)){
    geotiff_t* dem = new geotiff_t();
    dem->meta(path.c_str());
    images.push_back({
      path.string(),
      dem
    });
  }

  else for(const auto& entry: std::filesystem::directory_iterator(path)){
    geotiff_t* dem = new geotiff_t();
    dem->meta(entry.path().c_str());
    images.push_back({
      entry.path(),
      dem
    });
  }

 // Compute Bounding Box

  glm::vec2 min = glm::vec2(std::numeric_limits<value_t>::max());
  glm::vec2 max = glm::vec2(std::numeric_limits<value_t>::min());

  for(auto& image: images){
    min = glm::min(min, glm::vec2(image.tiff->min()));
    max = glm::max(max, glm::vec2(image.tiff->max()));
  }

  std::cout<<"Bounds: ";
  std::cout<<"("<<min.x<<" "<<min.y<<"), ";
  std::cout<<"("<<max.x<<" "<<max.y<<")"<<std::endl;

  // Construct Map  

  const glm::vec2 pscale = glm::vec2(1, 1); // In Meters
  const float hscale = 1.0f;
  glm::ivec2 mapdim = (max - min) / pscale;

  std::cout<<"Map: ";
  std::cout<<"("<<mapdim.x<<" "<<mapdim.y<<")"<<std::endl;

  world_t world(mapdim);
  for(auto [cell, pos]: world.map){
    cell.height = 0.0f;
  }

  // Load Images, Fill w. Height

  for(auto& image: images){

    image.tiff->read(image.name.c_str());

    // Tile Dimension in Map-Units
    const glm::vec2 tdim = image.tiff->scale() * image.tiff->dim() / pscale;
    const glm::vec2 tmin = (image.tiff->min() - min) / pscale;
    const glm::vec2 idim = image.tiff->dim();

    for(size_t tx = 0; tx < tdim.x; tx++)
    for(size_t ty = 0; ty < tdim.y; ty++){

      const glm::vec2 ipos = glm::vec2(tx, ty) * pscale / image.tiff->scale();
      const float value = image.tiff->operator[](glm::vec2(ipos.x, idim.y - 1 - ipos.y));

      glm::vec2 tpos = tmin + glm::vec2(tx, ty);
      tpos.y = mapdim.y - 1 - tpos.y;

      if(!world.map.oob(tpos))
        world.map.get(tpos)->height = value;

    }
  } 
  
  for(auto& image: images){
    delete image.tiff;
  }

  images.clear();

  float hmax = std::numeric_limits<value_t>::min();
  float hmin = std::numeric_limits<value_t>::max();

  for(auto [cell, pos]: world.map){
    if(cell.height < hmin) hmin = cell.height;
    if(cell.height > hmax) hmax = cell.height;
  }

  for(auto [cell, pos]: world.map){
    cell.height -= hmin;
  }

  for(auto [cell, pos]: world.map){
    cell.height /= hscale;
  }

  // Fill Normal
  for(auto [cell, pos]: world.map){
    cell.normal = soil::surface::normal(world, pos);
  }

  // Visualize Data

	Tiny::view.vsync = true;
	Tiny::window("soillib dataset viewer", 1200, 800);			//Open Window

	Tiny::event.handler = cam::handler;								//Event Handler
	Tiny::view.interface = [&](){ /* ... */ };				//No Interface

  Shader defaultShader({"shader/default.vs", "shader/default.fs"}, {"in_Position", "in_Normal"});

	Buffer positions, normals, indices;								//Define Buffers
	construct(world, positions, normals, indices);    //Fill Buffers

	Model mesh({"in_Position", "in_Normal"});					//Create Model with 2 Properties
  mesh.bind<glm::vec3>("in_Position", &positions);  //Bind Buffer to Property
  mesh.bind<glm::vec3>("in_Normal", &normals);      //Bind Buffer to Property
	mesh.index(&indices);

  // View Configuration

  cam::near = -100.0f;
  cam::far = 100.0f;
  cam::rot = 0.0f;
  cam::roty = 45.0f;
  cam::turnrate = 1.0f;
  cam::zoomrate *= 2.0f;
  cam::init(10, cam::ORTHO);
  cam::update();

  const float cheight = world.map.get(world.dim/2)->height;
  const glm::vec3 center = glm::vec3(-world.dim.x/2, -cheight, -world.dim.y/2);
  const glm::mat4 _scale = glm::scale(glm::mat4(1.0f), glm::vec3(100.0f/sqrt(world.dim.x*world.dim.y)));
  const glm::mat4 model = glm::translate(_scale, center);

  // Execute

	Tiny::view.pipeline = [&](){

		Tiny::view.target(color::white);        // Target Screen
		defaultShader.use();                    // Bind Shader
		defaultShader.uniform("model", model);  // Model Matrix
    defaultShader.uniform("vp", cam::vp);   // View-Projection Matrix
		mesh.render(GL_TRIANGLES);              // Render Model

	};

	Tiny::loop([](){});
	Tiny::quit();

	return 0;

}