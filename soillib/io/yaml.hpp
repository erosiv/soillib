#ifndef SOILLIB_IO_YAML
#define SOILLIB_IO_YAML

#include <soillib/external/mini-yaml/Yaml.hpp>

namespace soil {
namespace io {

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
