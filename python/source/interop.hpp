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
  soil::set(*target, source);

  // note: if the object comes in as a python object, we can tie the lifetime
  //  of the original object to the existence of the numpy object if the
  //  memory is shared and not copied. Note that here, we are copying!!!
  // owner = nb::find(soil::tensor source) // <- use find operator on
  //  object with original python pointer.

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
  soil::set(*target, source);

  switch(shape.dim){
    case 1: return __make_torch<T, 1>(target->data(), shape, owner);
    case 2: return __make_torch<T, 2>(target->data(), shape, owner);
    case 3: return __make_torch<T, 3>(target->data(), shape, owner);
    case 4: return __make_torch<T, 4>(target->data(), shape, owner);
    default: throw std::invalid_argument("too many dimensions");
  }

}

template<typename T>
soil::tensor __tensor_from_torch(const nb::ndarray<nb::pytorch>& array){

  const size_t size = array.size();
  T* data = (T*)array.data();
  
  const size_t ndim = array.ndim();
  const int d0 = (ndim >= 1) ? array.shape(0) : 1;
  const int d1 = (ndim >= 2) ? array.shape(1) : 1;
  const int d2 = (ndim >= 3) ? array.shape(2) : 1;
  const int d3 = (ndim >= 4) ? array.shape(3) : 1;
  auto shape = soil::shape(d0, d1, d2, d3);
  shape.dim = ndim;

  // Copy Data into New Tensor
  auto target_t = soil::tensor_t<T>(shape, soil::host_t::GPU);
  soil::set(target_t, soil::tensor_t<T>(data, shape, soil::host_t::GPU));
  return std::move(soil::tensor(target_t));

}