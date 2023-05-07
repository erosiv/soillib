
#include <soillib/map/basic.hpp>

#include <soillib/util/slice.hpp>
#include <soillib/util/pool.hpp>
#include <soillib/util/buf.hpp>
#include <soillib/util/index.hpp>

#include <soillib/soillib.hpp>

struct cell {
  float height = 0.0f;
};

soil::pool<cell> cellpool(1024*1024);
soil::map::basic<cell> map(cellpool);

int main(){

  // our map thereby exists, has pooled memory, and we can operate on it!

}
