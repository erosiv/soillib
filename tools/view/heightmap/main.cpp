#include <TinyEngine/TinyEngine>
#include <TinyEngine/color>
#include <TinyEngine/camera>
#include <TinyEngine/image>

#include <soillib/io/tiff.hpp>
#include <soillib/io/png.hpp>

#define GRIDSIZE 512

#include "model.h"



int main( int argc, char* args[] ) {

  if(argc < 2)
    return 0;

  std::string path = args[1];

  soil::io::tiff height((path + "height.tiff").c_str());
  soil::io::tiff discharge((path + "discharge.tiff").c_str());
  soil::io::png normal((path + "normal.png").c_str());

	for(int i = 0; i < dim.x; i++)
	for(int j = 0; j < dim.y; j++){
    	heightmap[i][j] = 80.0f*height[glm::ivec2(i, j)];
	}

	Tiny::view.vsync = false;
	Tiny::window("Heightmap Render", 1200, 800);			//Open Window

	cam::near = -500.0f;
	cam::far = 500.0f;
	cam::rot = 45.0f;
	cam::roty = 45.0f;
	cam::init(10, cam::ORTHO);
	cam::update();

	Tiny::event.handler = cam::handler;								//Event Handler
	Tiny::view.interface = [&](){ /* ... */ };				//No Interface

	Buffer positions, normals;												//Define Buffers
	Buffer indices;
	construct(positions, normals, indices);						//Call algorithm to fill buffers

	Model mesh({"in_Position", "in_Normal"});					//Create Model with 2 Properties
	mesh.bind<glm::vec3>("in_Position", &positions);	//Bind Buffer to Property
	mesh.bind<glm::vec3>("in_Normal", &normals);
	mesh.index(&indices);
	mesh.model = glm::translate(glm::mat4(1.0f), glm::vec3(-GRIDSIZE/2, -15.0, -GRIDSIZE/2));

	Shader defaultShader({"shader/default.vs", "shader/default.fs"}, {"in_Position", "in_Normal"});

  // Textures

  Texture dischargeMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(discharge[p]);
  }, glm::ivec2(GRIDSIZE)));

  Texture normalMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(normal[p]);
  }, glm::ivec2(GRIDSIZE)));

	Tiny::view.pipeline = [&](){											//Setup Drawing Pipeline

		Tiny::view.target(color::white);								//Target Screen

		defaultShader.use();														//Prepare Shader
		defaultShader.uniform("model", mesh.model);			//Set Model Matrix
    defaultShader.uniform("vp", cam::vp);						//View Projection Matrix
    defaultShader.texture("dischargeMap", dischargeMap);						//View Projection Matrix
    defaultShader.texture("normalMap", normalMap);						//View Projection Matrix
		mesh.render(GL_TRIANGLES);													//Render Model with Lines

	};

	Tiny::loop([&](){ //Autorotate Camera
	//	cam::pan(0.1f);
	});

	Tiny::quit();

	return 0;
}
