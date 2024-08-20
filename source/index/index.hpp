#ifndef SOILLIB_INDEX
#define SOILLIB_INDEX

#include <soillib/index/flat.hpp>

/*
The idea is that we have an index type, that allow for converting
coordinates to and from pure indices. This should basically replace
the currently existing shape type.

I am not quite sure what the optimal configuration of static and
dynamic typic is yet. In any case, this should effectively replace
the current implementation of the basic and quad maps.

How layer maps will work will be determined in the future.

The basic idea is to convert position -> index!
Index can be used to lookup values. This should also work for layers?
*/

/*
index.type() = flat?
index.type() = quad?
*/

namespace soil {

// this should be a wrapper around the actual implementation...
struct index {};

using shape = flat_t<2>;

 // flat index? multi-dimensional?
// non-compact quad index? basically that would be the quad map.
// 

} // end of namespace soil

#endif