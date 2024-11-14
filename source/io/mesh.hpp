#ifndef SOILLIB_IO_MESH
#define SOILLIB_IO_MESH

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <fstream>

namespace soil {
namespace io {

/*
Mesh File Structure:
- Export Only
- Load from Buffer, Index
- Generate List of Faces, Indices
- Option to Add Skirt (Or Not)
*/

struct mesh {

  mesh(){}

  // copy and move constructors...
  // construct directly from triangle and face data...
  // bounding box...

  mesh(const soil::buffer& _buffer, const soil::index& _index): _index{_index}, _buffer{_buffer}{

    // Get the Vertices...

    int ctot = 0;

    soil::select(_buffer.type(), [&]<std::floating_point T>(){
      auto buffer_t = _buffer.as<T>();

      const auto nan = std::numeric_limits<T>::quiet_NaN();

      soil::select(_index.type(), [&]<soil::index_2D I>(){
        auto index_t = _index.as<I>();

        for(auto pos: index_t.iter()){

          // I suppose we could just cheaply duplicate data...
          // and try to make it more efficient later...

          if(index_t.oob(pos + ivec2(0, 0))) continue;
          if(index_t.oob(pos + ivec2(0, 1))) continue;
          if(index_t.oob(pos + ivec2(1, 1))) continue;
          if(index_t.oob(pos + ivec2(1, 1))) continue;

          int i00 = index_t.flatten(pos + ivec2(0,0));
          int i01 = index_t.flatten(pos + ivec2(0,1));
          int i10 = index_t.flatten(pos + ivec2(1,0));
          int i11 = index_t.flatten(pos + ivec2(1,1));

          T v00 = buffer_t[i00];
          T v01 = buffer_t[i01];
          T v10 = buffer_t[i10];
          T v11 = buffer_t[i11];

          if(std::isnan(v00)) continue;
          if(std::isnan(v01)) continue;
          if(std::isnan(v10)) continue;
          if(std::isnan(v11)) continue;

          vec3 p00{ pos[0] + 0, pos[1] + 0, v00};
          vec3 p01{ pos[0] + 0, pos[1] + 1, v01};
          vec3 p10{ pos[0] + 1, pos[1] + 0, v10};
          vec3 p11{ pos[0] + 1, pos[1] + 1, v11};

          if(ctot == 0){
            this->min = p00;
            this->max = p00;
          } else {
            this->min = glm::min(this->min, p00);
            this->max = glm::max(this->max, p00);
          }

          const size_t count = this->vertices.size();

          ivec3 f0 = {count + 0, count + 1, count + 2};
          ivec3 f1 = {count + 1, count + 2, count + 3};

          this->vertices.push_back(p00);
          this->vertices.push_back(p01);
          this->vertices.push_back(p10);
          this->vertices.push_back(p11);

          this->faces.push_back(f0);
          this->faces.push_back(f1);

          ++ctot;

          // if(ctot > 64000)
          //   break;

        }
      });
    });

    std::cout<<"DONE"<<std::endl;

  }

  bool write(const char* filename) const;

private:
  std::vector<vec3> vertices;
  std::vector<ivec3> faces;

  vec3 min;
  vec3 max;

  soil::buffer _buffer;
  soil::index _index;
};

bool mesh::write(const char* filename) const {

  std::cout<<"WRITING"<<std::endl;

  std::ofstream out(filename, std::ios::out);
  if(!out){
    std::cout<<"Failed to open file "<<filename<<std::endl;
    return false;
  }

  // generate the vertex positions...

  out << "ply" << std::endl;
  out << "format ascii 1.0" << std::endl;
  out << "comment Created in soillib" << std::endl;
  out << "element vertex " << this->vertices.size() << std::endl;
  out << "property float x" << std::endl;
  out << "property float y" << std::endl;
  out << "property float z" << std::endl;
  out << "element face " << this->faces.size() << std::endl;
  out << "property list uchar uint vertex_indices" << std::endl;
  out << "end_header" << std::endl;

  for(auto& v: this->vertices){
    auto center = 0.5f*(this->max - this->min);
    auto vm = (v - this->min)/(this->max - this->min);// - center;
    out << vm[0] << " " << vm[1] << " " << vm[2] << std::endl;
  }

  for(auto& f: this->faces){
    out << 3 << " ";
    out << f[0] << " " << f[1] << " " << f[2] << std::endl;
  }

  out.close();
  return true;
}

} // end of namespace io
} // end of namespace soil

#endif