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

  typedef Yaml::Exception exception;
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
  
  template<typename T>
  T As() const {
    return n.As<T>();
  }

  yaml operator[](int i){
    auto sub = n[i];
    if(sub.IsNone()){
      throw exception("subnode does not exist", exception::OperationError);
    }
    return node(sub);
  }

  yaml operator[](const char* s){
    auto sub = n[s];
    if(sub.IsNone()){
      throw exception("subnode does not exist", exception::OperationError);
    }
    return node(sub);
  }

  yaml_iterator<pair> begin() const { 
    auto iter = n.Begin();
    if(iter.Type() == Yaml::ConstIterator::None)
      throw exception("can't iterate over node: empty", exception::OperationError);
    return yaml_iterator<pair>(iter); 
  }

  yaml_iterator<pair> end()   const { 
    auto iter = n.End();
    if(iter.Type() == Yaml::ConstIterator::None)
      throw exception("can't iterate over node: empty", exception::OperationError);
    return yaml_iterator<pair>(iter); 
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
    if(!is_key) throw yaml::exception("node index is not key", yaml::exception::OperationError);
    return key;
  }
  
  explicit operator size_t() const {
    if(is_key) throw yaml::exception("node index is not integer", yaml::exception::OperationError);
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

// Well-Known Types Pre-Define Parse Operators

void operator<<(glm::vec4& t, soil::io::yaml::node& node){
  t.x = node[0].As<float>();
  t.y = node[1].As<float>();
  t.z = node[2].As<float>();
  t.w = node[3].As<float>();
}

#endif
