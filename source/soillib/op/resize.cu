#include <soillib/soillib.hpp>

#include <soillib/op/resize.hpp>

namespace soil {

namespace {

inline int block(const int64_t elem, const int thread) {
  return int((elem + thread - 1) / thread);
}

GPU_ENABLE inline int iclamp(const int v, const int lo, const int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

}

//
// Bilinear Resample Kernel
//

template<size_t D>
__global__ void __resize (
  silt::tensor_t<float> tensorOut,      //!< Output Field
  const silt::tensor_t<float> tensorIn, //!< Input Field
  const silt::shape shapeOut,           //!< Output 2D Shape
  const silt::shape shapeIn             //!< Input 2D Shape
){

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shapeOut.elem()) return;

  using vec = silt::fvec<D>;
  const auto viewIn = tensorIn.view<vec>();
  auto viewOut = tensorOut.view<vec>();

  const silt::ivec2 opos = shapeOut.unflatten(n);

  // Map the output pixel center to a fractional position in input space.
  const float sx = float(shapeIn[0]) / float(shapeOut[0]);
  const float sy = float(shapeIn[1]) / float(shapeOut[1]);
  const float fx = (float(opos.x) + 0.5f) * sx - 0.5f;
  const float fy = (float(opos.y) + 0.5f) * sy - 0.5f;

  const int x0 = int(floorf(fx));
  const int y0 = int(floorf(fy));
  const float wx = fx - float(x0);
  const float wy = fy - float(y0);

  const int x0c = iclamp(x0,     0, shapeIn[0] - 1);
  const int x1c = iclamp(x0 + 1, 0, shapeIn[0] - 1);
  const int y0c = iclamp(y0,     0, shapeIn[1] - 1);
  const int y1c = iclamp(y0 + 1, 0, shapeIn[1] - 1);

  const vec v00 = viewIn[shapeIn.flatten(silt::ivec2(x0c, y0c))];
  const vec v10 = viewIn[shapeIn.flatten(silt::ivec2(x1c, y0c))];
  const vec v01 = viewIn[shapeIn.flatten(silt::ivec2(x0c, y1c))];
  const vec v11 = viewIn[shapeIn.flatten(silt::ivec2(x1c, y1c))];

  const vec v0 = v00 * (1.0f - wx) + v10 * wx;
  const vec v1 = v01 * (1.0f - wx) + v11 * wx;
  viewOut[n] = v0 * (1.0f - wy) + v1 * wy;

}

silt::tensor_t<float> resize(const silt::tensor_t<float>& tensor, const silt::shape shape_out) {

  const silt::shape shapeIn = tensor.shape();
  if (shapeIn.dim() < 2 || shape_out.dim() < 2)
    throw std::invalid_argument("resize requires 2D-indexed (or 2D + component) buffers");

  if (shapeIn[0] < 1 || shapeIn[1] < 1)
    throw std::invalid_argument("resize requires a non-empty input buffer");

  const int64_t comp = (shapeIn.dim() >= 3) ? shapeIn[2] : 1;

  const silt::shape shapeInFlat = silt::shape(shapeIn[0], shapeIn[1]);
  const silt::shape shapeOutFlat = silt::shape(shape_out[0], shape_out[1]);

  const silt::shape shapeOutFull = (comp == 1)
    ? silt::shape(shape_out[0], shape_out[1])
    : silt::shape(shape_out[0], shape_out[1], comp);

  auto output = silt::tensor_t<float>(shapeOutFull, silt::host_t::GPU);

  if(comp == 1) {
    __resize<1><<<block(shapeOutFlat.elem(), 512), 512>>>(output, tensor, shapeOutFlat, shapeInFlat);
    gpuErrchk(cudaGetLastError());
  } else if(comp == 2) {
    __resize<2><<<block(shapeOutFlat.elem(), 512), 512>>>(output, tensor, shapeOutFlat, shapeInFlat);
    gpuErrchk(cudaGetLastError());
  } else if(comp == 3) {
    __resize<3><<<block(shapeOutFlat.elem(), 512), 512>>>(output, tensor, shapeOutFlat, shapeInFlat);
    gpuErrchk(cudaGetLastError());
  } else {
    throw std::invalid_argument("resize only supports 1, 2 or 3 component buffers");
  }

  return output;

}

} // end of namespace soil
