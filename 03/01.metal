#include <metal_stdlib>

using namespace metal;

bool is_prime(uint n){
  if (n <= 1) return false;
  if (n <= 3) return true;
  if (n % 2 == 0 || n % 3 == 0) return false;
  for (uint i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) return false;
  }
  return true;
}

kernel void count_primes_non_atomic(device const uint* input [[ buffer(0) ]],
                                    device uint* prime_count       [[ buffer(1) ]],
                                    constant uint& N               [[ buffer(2) ]],
                                    uint tid                       [[ thread_position_in_grid ]],
                                    uint grid_size                 [[ threads_per_grid ]]) {

    for (uint i = tid; i < N; i += grid_size) {
        if (is_prime(input[i])) {
            prime_count[0]++;
        }
    }
}

kernel void count_primes_atomic(device const uint* input_array  [[ buffer(0) ]],
                                device atomic_uint* prime_count [[ buffer(1) ]],
                                constant uint& N                [[ buffer(2) ]],
                                uint tid                        [[ thread_position_in_grid ]],
                                uint grid_size                  [[ threads_per_grid ]])
{
    uint local_count = 0;
    for (uint i = tid; i < N; i += grid_size) {
        if (is_prime(input_array[i])) {
            local_count++;
        }
    }

    // Smanjujemo pritisak na memoriju: svaka dretva samo jednom radi atomsku operaciju na kraju
    if (local_count > 0) {
        atomic_fetch_add_explicit(prime_count, local_count, memory_order_relaxed);
    }
}
