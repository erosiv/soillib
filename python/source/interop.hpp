//
// Numpy Buffer from Type Buffer Generator
//

template<typename T, size_t D>
nb::object __make_numpy(T* data, const soil::shape shape, nb::capsule owner) {

  size_t _shape[D]{0};
  for(size_t d = 0; d < D; ++d)
    _shape[d] = shape[d];

  nb::ndarray<nb::numpy, T, nb::ndim<D>> array(
    data,
    D,
    _shape,
    owner
  );
  return nb::cast(std::move(array));

}

template<typename T>
nb::object __make_numpy(const soil::tensor_t<T>& source){

  const soil::shape shape = source.shape();
  soil::tensor_t<T>* target  = new soil::tensor_t<T>(shape, source.host()); 
  nb::capsule owner(target, [](void *p) noexcept {
    delete (soil::tensor_t<T>*)p;
  });
  soil::set(target->buffer(), source.buffer());

  switch(shape.dim){
    case 1: return __make_numpy<T, 1>(target->data(), shape, owner);
    case 2: return __make_numpy<T, 2>(target->data(), shape, owner);
    case 3: return __make_numpy<T, 3>(target->data(), shape, owner);
    case 4: return __make_numpy<T, 4>(target->data(), shape, owner);
    default: throw std::invalid_argument("too many dimensions");
  }

}

template<typename T>
nb::object __make_numpy(const soil::buffer_t<T>& source, soil::shape& shape){

  // Target Buffer (Heap)
  soil::buffer_t<T>* target  = new soil::buffer_t<T>(source.elem(), source.host()); 
  nb::capsule owner(target, [](void *p) noexcept {
    delete (soil::buffer_t<T>*)p;
  });
  soil::set(*target, source);

  switch(shape.dim){
    case 1: return __make_numpy<T, 1>(target->data(), shape, owner);
    case 2: return __make_numpy<T, 2>(target->data(), shape, owner);
    case 3: return __make_numpy<T, 3>(target->data(), shape, owner);
    case 4: return __make_numpy<T, 4>(target->data(), shape, owner);
    default: throw std::invalid_argument("too many dimensions");
  }

}

template<typename T>
soil::tensor __tensor_from_numpy(const nb::ndarray<nb::numpy>& array){

  const size_t size = array.size();
  const T* data = (T*)array.data();
  
  const size_t ndim = array.ndim();
  const int d0 = (ndim >= 1) ? array.shape(0) : 1;
  const int d1 = (ndim >= 2) ? array.shape(1) : 1;
  const int d2 = (ndim >= 3) ? array.shape(2) : 1;
  const int d3 = (ndim >= 4) ? array.shape(3) : 1;
  auto shape = soil::shape(d0, d1, d2, d3);
  shape.dim = ndim;

  auto tensor_t = soil::tensor_t<T>(shape, soil::host_t::CPU);
  for(size_t i = 0; i < size; ++i)
    tensor_t[i] = data[i];

  return std::move(soil::tensor(tensor_t));

}

//
// PyTorch Tensor Generation
//

template<typename T, size_t D>
nb::object __make_torch(T* data, const soil::shape shape, nb::capsule owner) {

  size_t _shape[D]{0};
  for(size_t d = 0; d < D; ++d)
    _shape[d] = shape[d];

  nb::ndarray<nb::pytorch, T, nb::ndim<D>> array(
    data,
    D,
    _shape,
    owner,
    nullptr,
    nb::dtype<T>(),
    nb::device::cuda::value
  );
  return nb::cast(std::move(array));

}

template<typename T>
nb::object __make_torch(const soil::tensor_t<T>& source){

  const soil::shape shape = source.shape();
  soil::tensor_t<T>* target  = new soil::tensor_t<T>(shape, source.host()); 
  nb::capsule owner(target, [](void *p) noexcept {
    delete (soil::tensor_t<T>*)p;
  });
  soil::set(target->buffer(), source.buffer());

  switch(shape.dim){
    case 1: return __make_torch<T, 1>(target->data(), shape, owner);
    case 2: return __make_torch<T, 2>(target->data(), shape, owner);
    case 3: return __make_torch<T, 3>(target->data(), shape, owner);
    case 4: return __make_torch<T, 4>(target->data(), shape, owner);
    default: throw std::invalid_argument("too many dimensions");
  }

}







template<typename T>
struct make_numpy {

  static nb::object operator()(soil::buffer& buffer){

  // buffer arrives as a python object. we tie the lifetime of the buffer to
  // the numpy array, so that the memory is always accessible / not deleted.

    soil::buffer_t<T> source = buffer.as<T>();

    if constexpr(nb::detail::is_ndarray_scalar_v<T>){

      size_t shape[1] = { source.elem() };
      nb::ndarray<nb::numpy, T, nb::ndim<1>> array(
        source.data(),    // raw data pointer
        1,                // number of dimensions
        shape,            // shape of array
        nb::find(buffer)  // lifetime guarantee
      );
      return nb::cast(std::move(array));

    } else {
      //! \todo add a concept to explicitly test for vector types
      
      constexpr int D = T::length();
      using V = soil::typedesc<T>::value_t;

      size_t shape[2] = { source.elem(), D };
      nb::ndarray<nb::numpy, V, nb::ndim<D>> array(
        source.data(),
        2,
        shape,
        nb::find(buffer)
      );
      return nb::cast(std::move(array));

    }

  }

};

//
// Torch Buffer Generator
//

template<typename T>
struct make_torch {

  static nb::object operator()(soil::buffer& buffer){

  // buffer arrives as a python object. we tie the lifetime of the buffer to
  // the numpy array, so that the memory is always accessible / not deleted.

    soil::buffer_t<T> source = buffer.as<T>();

    if constexpr(nb::detail::is_ndarray_scalar_v<T>){

      size_t shape[1] = { source.elem() };
      nb::ndarray<nb::pytorch, nb::device::cuda, T, nb::ndim<1>> array(
        source.data(),    // raw data pointer
        1,                // number of dimensions
        shape,            // shape of array
        nb::find(buffer), // lifetime guarantee
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));

    } else {
      //! \todo add a concept to explicitly test for vector types
      
      constexpr int D = T::length();
      using V = soil::typedesc<T>::value_t;
      
      size_t shape[2] = { source.elem(), D };
      nb::ndarray<nb::pytorch, nb::device::cuda, V, nb::ndim<D>> array(
        source.data(),
        2,
        shape,
        nb::find(buffer),
        nullptr,
        nb::dtype<V>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));

    } // else throw std::invalid_argument("can't convert non-scalar buffer type (yet)");

  }

  // note: this is generally bad, because it basically says that re-sampling
  //  can only happen with a CPU tensor type?
  //  so we really have to implement a re-sample kernel first.
  // ... so let's do that I guess???

  static nb::object operator()(soil::buffer& buffer, soil::shape& shape){

    if constexpr(nb::detail::is_ndarray_scalar_v<T>){

    soil::buffer_t<T> source = buffer.as<T>();
    soil::buffer_t<T>* target = new soil::buffer_t<T>(source.elem(), source.host());
    nb::capsule owner(target, [](void *p) noexcept {
      delete (soil::buffer_t<T>*)p;
    });
    soil::set(*target, source);

    size_t _shape[4]{0};
    for(size_t d = 0; d < 4; ++d)
      _shape[d] = shape[d];

    if(shape.dim == 1){
      nb::ndarray<nb::pytorch, T, nb::ndim<1>> array(
        target->data(),
        1,
        _shape,
        owner,
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));
    }
    else if(shape.dim == 2){
      nb::ndarray<nb::pytorch, T, nb::ndim<2>> array(
        target->data(),
        2,
        _shape,
        owner,
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));
    }
    else if(shape.dim == 3){
      nb::ndarray<nb::pytorch, T, nb::ndim<3>> array(
        target->data(),
        3,
        _shape,
        owner,
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));
    }
    else {
      nb::ndarray<nb::pytorch, T, nb::ndim<4>> array(
        target->data(),
        4,
        _shape,
        owner,
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));
    }

    }
    else throw std::invalid_argument("buffer type cannot be converted");

  }

};