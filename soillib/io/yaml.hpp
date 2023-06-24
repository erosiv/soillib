#ifndef SOILLIB_IO_YAML
#define SOILLIB_IO_YAML

#include <soillib/external/mini_yaml/Yaml.cpp>

namespace soil {
namespace io {

struct yaml {

    typedef Yaml::Node node;
    typedef Yaml::Exception exception;
    
    node root;

    yaml(std::string file){
        try {
            Yaml::Parse(root, file.c_str());
        } catch(Yaml::Exception e){
            isvalid = false;
        }
    }

    bool valid(){
        return isvalid;
    }


private:
    bool isvalid = true;

};

/*
extract nodes, then try and do what??

what if I have a single static yaml file or node,
then when something tries to load it can decide
whether to try and parse the yaml

lets start with map resolutions...

then images need to be able to take maps as an input directly
when filling, so that I don't need to pass the resolution!!!!
*/

// yaml loader: now or never

}; // end of namespace io
}; // end of namespace soil

#endif
