#include <TinyEngine/TinyEngine>
#include <TinyEngine/color>
#include <TinyEngine/camera>
#include <TinyEngine/image>

#include <soillib/soillib.hpp>
#include <soillib/io/yaml.hpp>
#include <soillib/io/tiff.hpp>
#include <soillib/io/png.hpp>

#include "model.hpp"

#include "source/world.hpp"
#include <iostream>
#include <csignal>

bool quit = false;

void sighandler(int signal){
  quit = true;
}

int main( int argc, char* args[] ) {

  // Load Configuration

  soil::io::yaml config("config.yaml");
  if(!config.valid()){
    std::cout<<"failed to load yaml configuration"<<std::endl;
  }

  try {
    World::config = config.As<world_c>();
  } catch(soil::io::yaml::exception e){
    std::cout<<"failed to parse yaml configuration: "<<e.what()<<std::endl; 
    return 0;
  }

  // Initialize World

  size_t SEED = time(NULL);
  if(argc >= 2)
    SEED = std::stoi(args[1]);

  World world(SEED);
  std::cout<<"SEED: "<<world.SEED<<std::endl;
  std::cout<<"DIM: "<<world.map.dimension.x<<" "<<world.map.dimension.y<<std::endl;

  // Setup Window

  Tiny::view.vsync = false;
  Tiny::window("soillib dataset viewer", 1200, 800);      //Open Window

  cam::near = -500.0f;
  cam::far = 500.0f;
  cam::rot = 45.0f;
  cam::roty = 45.0f;
  cam::turnrate = 0.1f;
  cam::zoomrate *= 0.1f;
  cam::init(2, cam::ORTHO);
  cam::update();

  bool paused = true;

  Tiny::event.handler = [&](){
    if(!Tiny::event.press.empty() && Tiny::event.press.back() == SDLK_p)
      paused = !paused;
    if(!Tiny::event.press.empty() && Tiny::event.press.back() == SDLK_m)
      cam::roty = 0.0f;
    cam::handler();               //Event Handler
  };

  Tiny::view.interface = [&](){ /* ... */ };        //No Interface

  Buffer positions, indices;                        //Define Buffers
  construct(world, positions, indices);               //Fill Buffers

  Model mesh({"in_Position"});          //Create Model with 2 Properties
  mesh.bind<glm::vec3>("in_Position", &positions);  //Bind Buffer to Property
  mesh.index(&indices);
  mesh.model = glm::translate(glm::mat4(1.0f), glm::vec3(-world.map.dimension.x/2, -15.0, -world.map.dimension.y/2));

  Shader defaultShader({"shader/default.vs", "shader/default.fs"}, {"in_Position"});

  // Run Erosion

 // signal(SIGINT, &sighandler);



  Texture dischargeMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(world.discharge(p));
  }, world.map.dimension));

  Texture normalMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(world.normal(p), 1.0f);
  }, world.map.dimension));

  Texture subNormalMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(world.subnormal(p), 1.0f);
  }, world.map.dimension));

  Texture albedoMap(image::make([&](const glm::ivec2 p){
    return glm::vec4(world.matrix(p).is_water?1.0f:0.0f);
  }, world.map.dimension));

  std::cout<<"AYY"<<std::endl;

  Tiny::view.pipeline = [&](){                      //Setup Drawing Pipeline

    Tiny::view.target(color::black);                //Target Screen

    defaultShader.use();                            //Prepare Shader
    defaultShader.uniform("model", mesh.model);     //Set Model Matrix
    defaultShader.uniform("vp", cam::vp);           //View Projection Matrix
    defaultShader.texture("dischargeMap", dischargeMap);            //View Projection Matrix
    defaultShader.texture("normalMap", normalMap);            //View Projection Matrix
    defaultShader.texture("subNormalMap", subNormalMap);            //View Projection Matrix
    defaultShader.texture("albedoMap", albedoMap);            //View Projection Matrix
  //  defaultShader.uniform("albedoRead", albedo_read);            //View Projection Matrix
    defaultShader.uniform("dimension", glm::vec2(world.map.dimension));           //View Projection Matrix
    mesh.render(GL_TRIANGLES);                          //Render Model with Lines

  };

  Tiny::loop([&](){ //Autorotate Camera
    if(paused)
      return;

    world.erode();

    dischargeMap.raw(image::make([&](const glm::ivec2 p){
      return glm::vec4(world.discharge(p));
    }, world.map.dimension));

    normalMap.raw(image::make([&](const glm::ivec2 p){
      return glm::vec4(world.normal(p), 0.0f);
    }, world.map.dimension));

    subNormalMap.raw(image::make([&](const glm::ivec2 p){
        return glm::vec4(world.subnormal(p), 1.0f);
      }, world.map.dimension));

    albedoMap.raw(image::make([&](const glm::ivec2 p){
      return glm::vec4(world.matrix(p).is_water?1.0f:0.0f);
    }, world.map.dimension));

    construct(world, positions, indices);               //Fill Buffers

  });

  Tiny::quit();

  // Export Images

  soil::io::tiff discharge(world.map.dimension);
  discharge.fill([&](const glm::ivec2 pos){
    return world.discharge(pos);
  });
  discharge.write("out/discharge.tiff");

  soil::io::tiff height(world.map.dimension);
  height.fill([&](const glm::ivec2 pos){
    return world.map.top(pos)->height;
  });
  height.write("out/height.tiff");

  soil::io::tiff subheight(world.map.dimension);
  subheight.fill([&](const glm::ivec2 pos){
    return world.subheight(pos)/world.config.scale;
  });
  subheight.write("out/subheight.tiff");

  soil::io::tiff vegetation(world.map.dimension);
  vegetation.fill([&](const glm::ivec2 pos){
    return world.map.get(pos)->rootdensity;
  });
  vegetation.write("out/vegetation.tiff");

  soil::io::png normal(world.map.dimension);
  normal.fill([&](const glm::ivec2 pos){
    glm::vec3 normal = world.normal(pos);
    normal = glm::vec3(normal.x, -normal.z, normal.y);
    normal = 0.5f*normal + 0.5f;
    return 255.0f*glm::vec4(normal, 1.0f);
  });
  normal.write("out/normal.png");

  soil::io::png subnormal(world.map.dimension);
  subnormal.fill([&](const glm::ivec2 pos){
    glm::vec3 normal = world.subnormal(pos);
    normal = glm::vec3(normal.x, -normal.z, normal.y);
    normal = 0.5f*normal + 0.5f;
    return 255.0f*glm::vec4(normal, 1.0f);
  });
  subnormal.write("out/subnormal.png");

  soil::io::png albedo(world.map.dimension);
  albedo.fill([&](const glm::ivec2 pos){
    float type = world.matrix(pos).is_water?0.5f:1.0f;
    return 255.0f*glm::vec4(glm::mix(glm::vec3(0,0,0), glm::vec3(1,1,1), type), 1.0f);
  });
  albedo.write("out/albedo.png");

  return 0;
}
