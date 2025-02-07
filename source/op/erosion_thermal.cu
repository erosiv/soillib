namespace soil {


//
// Cascading Kernel
//
// Effectively we have to compute the height-difference between every cell
// and its non-out-of-bounds neighbors, then we have to transfer the sediment.
// How do we do this without race conditions?
// I suppose that we need an additional buffer to determine the updated sediment amounts...
// so that we can ping-pong back and forth...

// for now, we will implement this as a device function locally and perhaps switch to
// a singular kernel later.

// __global__ void cascade(model_t model){
// }

__device__ void cascade(model_t& model, const glm::ivec2 ipos, buffer_t<float>& transfer, const param_t param) {

  if(model.index.oob(ipos))
    return;

  // Get Non-Out-of-Bounds Neighbors

  const glm::ivec2 n[] = {
    glm::ivec2(-1, -1),
    glm::ivec2(-1, 0),
    glm::ivec2(-1, 1),
    glm::ivec2(0, -1),
    glm::ivec2(0, 1),
    glm::ivec2(1, -1),
    glm::ivec2(1, 0),
    glm::ivec2(1, 1)
  };

  struct Point {
    glm::ivec2 pos;
    float h;
    float d;
  } sn[8];

  int num = 0;

  for(auto &nn : n){

    glm::ivec2 npos = ipos + nn;

    if(model.index.oob(npos))
      continue;

    const size_t index = model.index.flatten(npos);
    const float height = model.height[index];
    sn[num] = {npos, height, length(glm::vec2(nn))};
    ++num;
  }

  const size_t index = model.index.flatten(ipos);
  const float height = model.height[index];
  float h_ave = height;
  // for (int i = 0; i < num; ++i)
  //   h_ave += sn[i].h;
  // h_ave /= (float)(num + 1);

  float transfer_tot = 0.0f;
  
  for(int i = 0; i < num; ++i){

    // Full Height-Different Between Positions!
    float diff = h_ave - sn[i].h;
    if (diff == 0) // No Height Difference
      continue;

    // The Amount of Excess Difference!
    float excess = 0.0f;
    excess = abs(diff) - sn[i].d * param.maxdiff;
    if (excess <= 0) // No Excess
      continue;

    excess = (diff > 0) ? -excess : excess;

    // Actual Amount Transferred
    float transfer = param.settling * excess / 2.0f;
    transfer_tot += transfer;
  }

  transfer[index] = transfer_tot / (float) num;
}

__global__ void compute_cascade(model_t model, buffer_t<float> transfer, const param_t param){

  // note: this should be two kernels.
  // one to compute the amount of sediment
  // that has to be transferred -
  // one to then actual transfer it.

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;
  const ivec2 ipos = model.index.unflatten(ind);
  cascade(model, ipos, transfer, param);

}

__global__ void apply_cascade(model_t model, buffer_t<float> transfer_b, const param_t param){
  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;

  // Only cascade where agitation exists?

  const float transfer = transfer_b[ind];
  const float discharge = log(1.0f + model.discharge[ind])/6.0f;
  model.height[ind] += discharge * transfer;
  //model.height[ind] += transfer;
}

}