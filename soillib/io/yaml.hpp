#ifndef SOILLIB_IO_YAML
#define SOILLIB_IO_YAML

#include <soillib/external/mini_yaml/Yaml.cpp>

namespace soil {
namespace io {

template<typename T>
struct yaml_iterator {

  yaml_iterator(Yaml::ConstIterator iter) noexcept: iter(iter) {}

  const yaml_iterator& operator++() noexcept {
    iter++;
    ind++;
    return *this;
  };

  const bool operator!=(yaml_iterator other) noexcept {
    return this->iter != other.iter;
  };

  T operator*() noexcept {
    if(iter.Type() == Yaml::ConstIterator::MapType)
      return T((*iter).first, (*iter).second);
    else return T(ind, (*iter).second);
  };

private:

  Yaml::ConstIterator iter;
  size_t ind = 0;

};

struct yaml_pair;
struct yaml {

  typedef Yaml::OperationException exception;
  typedef Yaml::Node node;
  typedef yaml_pair pair;

  yaml(){}
  yaml(const node n):n(n){}
  yaml(const char* filename){
   try {
      Yaml::Parse(n, filename);
    } catch(exception e){
      isvalid = false;
    }
  }

  bool valid(){
    return isvalid;
  }

  // Indexing

  yaml operator[](int i){
    auto sub = n[i];
    if(sub.IsNone()){
      throw exception("subnode does not exist");
    }
    return node(sub);
  }

  yaml operator[](const char* s){
    auto sub = n[s];
    if(sub.IsNone()){
      throw exception("subnode does not exist");
    }
    return node(sub);
  }

  // Iteration

  yaml_iterator<pair> begin() const { 
    auto iter = n.Begin();
    if(iter.Type() == Yaml::ConstIterator::None)
      throw exception("can't iterate over node: empty");
    return yaml_iterator<pair>(iter); 
  }

  yaml_iterator<pair> end()   const { 
    auto iter = n.End();
    if(iter.Type() == Yaml::ConstIterator::None)
      throw exception("can't iterate over node: empty");
    return yaml_iterator<pair>(iter); 
  }

  // Casting

  // Note: Either define As() function OR cast struct

  template<typename T>
  struct cast {
    static T As(const yaml& node){
      return node.n.As<T>();
    }
  };
  
  template<typename T>
  T As() {
    return cast<T>::As(*this);
  }

private:

  node n;
  bool isvalid = true;

};

// Index or Key Type

struct indkey {

  bool is_key = true;

  indkey(std::string key):key(key),is_key(true){}
  indkey(size_t ind):ind(ind),is_key(false){}

  std::string key;
  size_t ind;
  
  explicit operator std::string() const {
    if(!is_key) throw yaml::exception("node index is not key");
    return key;
  }
  
  explicit operator size_t() const {
    if(is_key) throw yaml::exception("node index is not integer");
    return ind;
  }

};

struct yaml_pair {
  indkey ik;
  yaml n;
};

std::ostream& operator<<(std::ostream& os, indkey& ik){
  if(ik.is_key) 
      return os << ik.key;
  else 
    return os << ik.ind;
}

}; // end of namespace io
}; // end of namespace soil

// Well-Known Types with Pre-Defined Cast Operators

// GLM-Types

template<typename T>
struct soil::io::yaml::cast<glm::tvec2<T>> {
  static glm::tvec2<T> As(soil::io::yaml& node){
    return glm::tvec2<T> {
      node[0].As<T>(),
      node[1].As<T>()
    };
  }
};

template<typename T>
struct soil::io::yaml::cast<glm::tvec3<T>> {
  static glm::tvec3<T> As(soil::io::yaml& node){
    return glm::tvec3<T> {
      node[0].As<T>(),
      node[1].As<T>(),
      node[2].As<T>()
    };
  }
};

template<typename T>
struct soil::io::yaml::cast<glm::tvec4<T>> {
  static glm::tvec4<T> As(soil::io::yaml& node){
    return glm::tvec4<T> {
      node[0].As<T>(),
      node[1].As<T>(),
      node[2].As<T>(),
      node[3].As<T>()
    };
  }
};

// STL Containers
//  Note: Only those required by Mini-Yaml

template<typename T>
struct soil::io::yaml::cast<std::vector<T>> {
  static std::vector<T> As(soil::io::yaml& node){
    std::vector<T> vector;
    for(auto [key, sub]: node)
      vector.push_back(sub.As<T>());
    return vector;
  }
};

template<typename T>
struct soil::io::yaml::cast<std::map<size_t, T>> {
  static std::map<size_t, T> As(soil::io::yaml& node){
    std::map<size_t, T> map;
    for(auto [key, sub]: node)
      map.emplace((size_t)key, sub.As<T>());
    return map;
  }
};

template<typename T>
struct soil::io::yaml::cast<std::map<std::string, T>> {
  static std::map<std::string, T> As(soil::io::yaml& node){
    std::map<std::string, T> map;
    for(auto [key, sub]: node)
      map.emplace((std::string)key, sub.As<T>());
    return map;
  }
};

#endif
