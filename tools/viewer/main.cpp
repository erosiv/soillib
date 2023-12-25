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

  std::string path = args[1];

  // Load Image Data, Create Map

  soil::io::geotiff dem(path.c_str());
  world_t world(dem.dim());

  // Fill Map

  for(auto [cell, pos]: world.map)
    cell.height = dem[pos];

  for(auto [cell, pos]: world.map)
    cell.normal = soil::surface::normal(world, pos);

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
  const glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(100.0f/sqrt(world.dim.x*world.dim.y)));
  const glm::mat4 model = glm::translate(scale, center);

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