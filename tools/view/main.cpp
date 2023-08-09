#include <TinyEngine/TinyEngine>
#include <TinyEngine/color>
#include <TinyEngine/camera>
#include <TinyEngine/image>

#include <soillib/util/pool.hpp>
#include <soillib/util/index.hpp>
#include <soillib/map/basic.hpp>
#include <soillib/io/tiff.hpp>
#include <soillib/io/png.hpp>

#include "model.hpp"

struct cell {
  float height;
  float discharge;
  glm::vec4 normal;
};

int main( int argc, char* args[] ) {

  if(argc < 2){
    std::cout<<"please specify input directory for dataset"<<std::endl;
    return 0;
  }
  std::string path = args[1];

  // Load Image Data

  soil::io::tiff height((path + "/height.tiff").c_str());
  soil::io::tiff discharge((path + "/discharge.tiff").c_str());
  soil::io::png normal((path + "/normal.png").c_str());
  soil::io::png albedo((path + "/albedo.png").c_str());

  // Create Map

  const glm::ivec2 dim = glm::ivec2(height.width, height.height);
  soil::map::basic<cell> map(dim);

  // Fill Map

  for(auto [cell, pos]: map){
    cell.height = 80.0f*height[pos];
    cell.discharge = discharge[pos];
    cell.normal = glm::vec4(normal[pos]);
  }

	Tiny::view.vsync = false;
	Tiny::window("soillib dataset viewer", 1200, 800);			//Open Window

	cam::near = -500.0f;
	cam::far = 500.0f;
	cam::rot = 45.0f;
	cam::roty = 45.0f;
  cam::turnrate = 0.1f;
	cam::init(10, cam::ORTHO);
	cam::update();

	Tiny::event.handler = cam::handler;								//Event Handler
	Tiny::view.interface = [&](){ /* ... */ };				//No Interface

	Buffer positions, indices;												//Define Buffers
	construct(map, positions, indices);						    //Fill Buffers

	Model mesh({"in_Position"});					//Create Model with 2 Properties
	mesh.bind<glm::vec3>("in_Position", &positions);	//Bind Buffer to Property
	mesh.index(&indices);
	mesh.model = glm::translate(glm::mat4(1.0f), glm::vec3(-map.dimension.x/2, -15.0, -map.dimension.y/2));

	Shader defaultShader({"shader/default.vs", "shader/default.fs"}, {"in_Position"});

  // Textures

  Texture dischargeMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(discharge[p]);
  }, map.dimension));

  Texture normalMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(normal[p]);
  }, map.dimension));

  Texture albedoMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(albedo[p]);
  }, map.dimension));

	Tiny::view.pipeline = [&](){											//Setup Drawing Pipeline

		Tiny::view.target(color::white);								//Target Screen

		defaultShader.use();														//Prepare Shader
		defaultShader.uniform("model", mesh.model);			//Set Model Matrix
    defaultShader.uniform("vp", cam::vp);						//View Projection Matrix
    defaultShader.texture("dischargeMap", dischargeMap);						//View Projection Matrix
    defaultShader.texture("normalMap", normalMap);            //View Projection Matrix
    defaultShader.texture("albedoMap", albedoMap);            //View Projection Matrix
    defaultShader.uniform("dimension", glm::vec2(map.dimension));						//View Projection Matrix
		mesh.render(GL_TRIANGLES);													//Render Model with Lines

	};

	Tiny::loop([&](){ //Autorotate Camera
	//	cam::pan(0.1f);
	});

	Tiny::quit();

	return 0;
}
