bool is_prime(int n) {
  if (n <= 1) {
    return false;
  }
  if (n <= 3) {
    return true;
  }
  if (n % 2 == 0 || n % 3 == 0) {
    return false;
  }
  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) {
      return false;
    }
  }
  return true;
}

__kernel void count_primes_non_atomic(__global const int *input, __global int *result,
                                      const int N) {
  uint tid = get_global_id(0);
  uint grid_size = get_global_size(0);

  for (int i = tid; i < N; i += grid_size) {
    if (is_prime(input[i])) {
      ++result[0];
    }
  }
}

__kernel void count_primes_atomic(__global const int *input, __global int *result,
                                  const int N) {
  int tid = get_global_id(0);
  int grid_size = get_global_size(0);
  int local_count = 0;

  for (int i = tid; i < N; i += grid_size) {
    if (is_prime(input[i])) {
      ++local_count;
    }
  }

  if (local_count > 0) {
    atomic_add(result, local_count);
  }
}
