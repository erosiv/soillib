#ifndef SOILLIB_IO_MESH
#define SOILLIB_IO_MESH

#include <fstream>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <unordered_map>

namespace soil {
namespace io {

namespace {

constexpr bool isBigEndianArchitecture() {
  constexpr uint32_t i = 0x01020304;
  return reinterpret_cast<const uint8_t *>(&i)[0] == 1;
}

} // namespace

//! Mesh is a simple data-oriented triangle mesh struct
//!
//! Mesh can be constructed directly from a buffer and
//! 2D index, creating a simple terrain triangle mesh.
//!
//! The mesh is optimized for compactness, and there is
//! an option to add a skirt to the terrain mesh.
//!
//! The export format is a binary .ply file.
//!
struct mesh {

  mesh() {}
  mesh(const soil::buffer &_buffer, const soil::index &_index, const vec3 scale) {

    this->triangulate(_buffer, _index, scale);

    // Compute Min, Max
    this->min = vec3(std::numeric_limits<float>::max());
    this->max = vec3(std::numeric_limits<float>::min());
    for (auto v : this->vertices) {
      this->min = glm::min(this->min, v);
      this->max = glm::max(this->max, v);
    }
  }

  void triangulate(const soil::buffer &buffer, const soil::index &index, const vec3 scale) {

    soil::select(buffer.type(), [&]<std::floating_point T>() {
      auto buffer_t = buffer.as<T>();

      soil::select(index.type(), [&]<soil::index_2D I>() {
        auto index_t = index.as<I>();

        // Insert Vertices:
        //  Construct Map from Buffer Index to Mesh Index
        //  Insert Vertices
        //  Use Map to Insert Face Triangles

        std::unordered_map<int, unsigned int> vertex_set;

        // Insert Vertices
        unsigned int count = 0;
        for (auto pos : index_t.iter()) {

          int ind = index_t.flatten(pos); // Buffer Position Index
          T val = buffer_t[ind];          // Buffer Value
          if (std::isnan(val))            // Non NaN Values!
            continue;

          vertex_set.insert({ind, count}); // Map from Buffer to Mesh
          vec3 p(pos[0], pos[1], val);
          p /= scale; // Scale Position
          this->vertices.push_back(p);
          ++count;
        }

        // Insert Faces
        for (auto pos : index_t.iter()) {

          if (index_t.oob(pos + ivec2(0, 0)))
            continue;
          if (index_t.oob(pos + ivec2(0, 1)))
            continue;
          if (index_t.oob(pos + ivec2(1, 1)))
            continue;
          if (index_t.oob(pos + ivec2(1, 1)))
            continue;

          int i00 = index_t.flatten(pos + ivec2(0, 0));
          int i01 = index_t.flatten(pos + ivec2(0, 1));
          int i10 = index_t.flatten(pos + ivec2(1, 0));
          int i11 = index_t.flatten(pos + ivec2(1, 1));

          T v00 = buffer_t[i00];
          T v01 = buffer_t[i01];
          T v10 = buffer_t[i10];
          T v11 = buffer_t[i11];

          if (std::isnan(v00))
            continue;
          if (std::isnan(v01))
            continue;
          if (std::isnan(v10))
            continue;
          if (std::isnan(v11))
            continue;

          uvec3 f0 = {vertex_set[i01], vertex_set[i00], vertex_set[i10]};
          uvec3 f1 = {vertex_set[i01], vertex_set[i10], vertex_set[i11]};

          this->faces.push_back(f0);
          this->faces.push_back(f1);
        }
      });
    });
  }

  void center() {
    auto center = 0.5f * (this->max + this->min);
    for (auto &v : this->vertices) {
      v -= center;
    }
  }

  bool write(const char *filename) const;
  bool write_binary(const char *filename) const;

private:
  std::vector<vec3> vertices; //!< Type: float
  std::vector<uvec3> faces;   //!< Type: unsigned int
  vec3 min;
  vec3 max;
};

bool mesh::write(const char *filename) const {

  std::ofstream out(filename, std::ios::out);
  if (!out) {
    std::cout << "Failed to open file " << filename << std::endl;
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

  for (auto &v : this->vertices) {
    auto center = 0.5f * (this->max - this->min);
    auto vm = (v - this->min) / (this->max - this->min); // - center;
    out << vm[0] << " " << vm[1] << " " << vm[2] << std::endl;
  }

  for (auto &f : this->faces) {
    out << 3 << " ";
    out << f[0] << " " << f[1] << " " << f[2] << std::endl;
  }

  out.close();
  return true;
}

bool mesh::write_binary(const char *filename) const {

  std::ofstream fout(filename, std::ios::binary);
  if (!fout) {
    std::cout << "Failed to open file " << filename << std::endl;
    return false;
  }

  // Write header
  fout << "ply\n";
  if (isBigEndianArchitecture())
    fout << "format binary_big_endian 1.0\n";
  else
    fout << "format binary_little_endian 1.0\n";

  fout << "element " << "vertex" << " " << this->vertices.size() << std::endl;
  fout << "property float x" << std::endl;
  fout << "property float y" << std::endl;
  fout << "property float z" << std::endl;
  fout << "element " << "face" << " " << this->faces.size() << std::endl;
  fout << "property list uchar uint vertex_indices" << std::endl;
  fout << "end_header" << std::endl;

  for (auto &v : this->vertices) {
    fout.write(reinterpret_cast<const char *>(&v[0]), 3 * sizeof(float));
  }

  for (auto &f : this->faces) {
    const unsigned char count = 3;
    fout.write(reinterpret_cast<const char *>(&count), sizeof(unsigned char));
    glm::vec<3, unsigned int, glm::packed_highp> fv = f;
    fout.write(reinterpret_cast<const char *>(&fv[0]), 3 * sizeof(unsigned int));
  }

  fout.close();
  return true;
}

} // end of namespace io
} // end of namespace soil

#endif