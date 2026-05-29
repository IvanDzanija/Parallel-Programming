// https://github.com/maoshouse/OpenCL-reduction-sum/blob/master/sum.cl
__kernel void sum_reduction(__global const float *input, __global float *output,
                            const int input_size, __local float *reduction_sums) {
  const int gid = get_global_id(0);
  const int lid = get_local_id(0);
  const int local_size = get_local_size(0);
  const int group_id = get_group_id(0);

  reduction_sums[lid] = input[gid];

  for (int offset = local_size / 2; offset > 0; offset /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);  // wait for all other work-items to finish previous
                                   // iteration.
    if (lid < offset) {
      reduction_sums[lid] += reduction_sums[lid + offset];
    }
  }

  if (lid == 0) {
    output[group_id] = reduction_sums[0];
  }
}

__kernel void partial_pi(const int N, __global float *result) {
  int tid = get_global_id(0);
  int grid_size = get_global_size(0);
  float total = 0;

  for (int i = tid; i < N; i += grid_size) {
    float current = ((i - 0.5) / N) * ((i - 0.5) / N);
    total += 1.0 / (1.0 + current);
  }
  result[tid] = total;
}
