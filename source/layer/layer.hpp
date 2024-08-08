#ifndef SOILLIB_LAYER
#define SOILLIB_LAYER

#include <soillib/soillib.hpp>

/*
Layer Concept:

Like in GIMP, or Photoshop, a full model consists of multiple layers of information.
This name might be subject to change, since they are not necessarily "vertically layered".

The idea is that these act as generic maps, which take the data from an array or another
spatially organizing structure and allow for computing maps directly, caching results and
returning fully pre-computed arrays on request.

We will try to implement this for now and later refine the concept to be more modular.
This can later also be used to have a node-based synthesis model, but primarily exists
so that we can define common operations and even one day autograd through them.
*/

/*
Implementation: Does the layer have a strict-typed input and output? I think so.
Does it store a reference to its underlying in type?

In theory, I could have a generic node that just takes a lambda right?
Let's not do that for now though. In theory this could be done later
for proper interactive nodes w. std::function?
*/

namespace soil {

template<typename Tin, typename Tout>
struct layer {

  typedef Tin in_t;   // Incoming Data-Type
  typedef Tout out_t; // Outgoing Data-Type

  layer(Tin& in):
    in{in}{}

protected:
  in_t& in;
};

} // end of namespace soil

#endif