__kernel void partial_pi(const int N, __global double *result) {
  int tid = get_global_id(0);
  int grid_size = get_global_size(0);
  int total = 0;

  for (int i = tid; i < N; i += grid_size) {
    int current = ((i - 0.5) / N) * ((i - 0.5) / N);
    total += 1.0 / (1.0 + current);
  }
  if (total > 0) {
    atomic_add(&result, total);
  }
}
