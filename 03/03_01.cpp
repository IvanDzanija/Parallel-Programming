#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined _MSC_VER
#pragma comment(lib, "OpenCL.lib")
#define NOTIFY_CONVENTION __stdcall
#else
#define NOTIFY_CONVENTION
#endif

#include <OpenCL/opencl.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
// #include <unistd.h>

#define CL_CHECK(_expr)                                                      \
  do {                                                                       \
    cl_int _err = _expr;                                                     \
    if (_err == CL_SUCCESS) break;                                           \
    fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
    abort();                                                                 \
  } while (0)

void *CL_CHECK_ERR(void *_val) {
  if (_val != NULL) {
    return _val;
  }
  fprintf(stderr, "OpenCL Error\n");
  abort();
  return NULL;
}

void NOTIFY_CONVENTION pfn_notify(const char *errinfo, const void *private_info,
                                  size_t cb, void *user_data) {
  fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

const int LIMIT = 100000;
std::vector<bool> is_prime(LIMIT + 1, true);

void generate_primes() {
  // Sieve
  is_prime[0] = is_prime[1] = false;
  for (int i = 2; i <= LIMIT; i++) {
    if (is_prime[i] && (long long)i * i <= LIMIT) {
      for (int j = i * i; j <= LIMIT; j += i) {
        is_prime[j] = false;
      }
    }
  }
}

int main(int argc, char **argv) {
  generate_primes();
  // OpenCL setup
  // Find platform
  cl_platform_id platform;
  CL_CHECK(clGetPlatformIDs(1, &platform, NULL));
  // Find device
  cl_device_id device;
  CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
  // Create context
  cl_int _err = CL_INVALID_VALUE;
  cl_context context = (cl_context)CL_CHECK_ERR(
      clCreateContext(NULL, 1, &device, &pfn_notify, NULL, &_err));
  // Create command queue
  cl_command_queue queue = (cl_command_queue)CL_CHECK_ERR(
      clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &_err));

  // Load kernel source
  std::string current_line;
  std::string source;
  std::string file_name = "primes.cl";
  std::ifstream kernel_file(file_name);
  while (std::getline(kernel_file, current_line)) {
    source += current_line + "\n";
  }

  // Create program
  const char *source_c_str = source.c_str();
  cl_program program = (cl_program)CL_CHECK_ERR(
      clCreateProgramWithSource(context, 1, &source_c_str, NULL, &_err));

  // Build program
  if (clBuildProgram(program, 1, &device, "", NULL, NULL) != CL_SUCCESS) {
    char buffer[10240];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer,
                          NULL);
    fprintf(stderr, "CL Compilation failed:\n%s", buffer);
    abort();
  }

  // Create kernels
  cl_kernel atomic_kernel =
      (cl_kernel)CL_CHECK_ERR(clCreateKernel(program, "count_primes_atomic", &_err));
  cl_kernel non_atomic_kernel = (cl_kernel)CL_CHECK_ERR(
      clCreateKernel(program, "count_primes_non_atomic", &_err));

  // Events for measurement
  cl_event event_write;
  cl_event atomic_event_read;
  cl_event non_atomic_event_read;
  cl_event atomic_event_kernel;
  cl_event non_atomic_event_kernel;

  const size_t N = (1 << 24);  // 2^20
  std::vector<int> input(N);
  int atomic_result = -1;
  int non_atomic_result = -1;
  srand((unsigned int)time(NULL));
  for (size_t i = 0; i < N; ++i) {
    input[i] = (rand() % LIMIT) + 1;
  }

  // Create buffers
  cl_mem input_buffer = (cl_mem)CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N,
                     input.data(), &_err));
  CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(int) * N,
                                input.data(), 0, NULL, &event_write));

  // Result buffer (reduced to a single integer)
  cl_mem result_buffer = (cl_mem)CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &_err));

  // Set kernel arguments
  CL_CHECK(clSetKernelArg(atomic_kernel, 0, sizeof(cl_mem), &input_buffer));
  CL_CHECK(clSetKernelArg(atomic_kernel, 1, sizeof(cl_mem), &result_buffer));
  CL_CHECK(clSetKernelArg(atomic_kernel, 2, sizeof(int), &N));
  CL_CHECK(clSetKernelArg(non_atomic_kernel, 0, sizeof(cl_mem), &input_buffer));
  CL_CHECK(clSetKernelArg(non_atomic_kernel, 1, sizeof(cl_mem), &result_buffer));
  CL_CHECK(clSetKernelArg(non_atomic_kernel, 2, sizeof(int), &N));

  // Execute kernel and read result
  size_t Ls[] = {32, 64, 128, 256};      // Threads per block
  size_t Bs[] = {16, 32, 64, 128, 256};  // Number of blocks
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 5; ++j) {
      size_t L = Ls[i];
      size_t B = Bs[j];
      size_t G = B * L;
      size_t global_work_size[1] = {G};
      size_t local_work_size[1] = {L};
      int zero = 0;
      CL_CHECK(clEnqueueWriteBuffer(queue, result_buffer, CL_TRUE, 0, sizeof(int),
                                    &zero, 0, NULL, NULL));
      CL_CHECK(clEnqueueNDRangeKernel(queue, atomic_kernel, 1, NULL, global_work_size,
                                      local_work_size, 0, NULL, &atomic_event_kernel));
      CL_CHECK(clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, sizeof(int),
                                   &atomic_result, 0, NULL, &atomic_event_read));

      // Reset result buffer for the second kernel
      CL_CHECK(clEnqueueWriteBuffer(queue, result_buffer, CL_TRUE, 0, sizeof(int),
                                    &zero, 0, NULL, NULL));

      // Execute non-atomic kernel and read result
      CL_CHECK(clEnqueueNDRangeKernel(queue, non_atomic_kernel, 1, NULL,
                                      global_work_size, local_work_size, 0, NULL,
                                      &non_atomic_event_kernel));
      CL_CHECK(clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, sizeof(int),
                                   &non_atomic_result, 0, NULL,
                                   &non_atomic_event_read));

      // Asynchronous finish
      CL_CHECK(clFinish(queue));

      cl_ulong start, end;
      double time_write = 0;
      double atomic_time_kernel = 0;
      double non_atomic_time_kernel = 0;
      double atomic_time_read = 0;
      double non_atomic_time_read = 0;

      // Write input buffer
      clGetEventProfilingInfo(event_write, CL_PROFILING_COMMAND_START, sizeof(start),
                              &start, NULL);
      clGetEventProfilingInfo(event_write, CL_PROFILING_COMMAND_END, sizeof(end), &end,
                              NULL);
      time_write += (double)(end - start) * 1e-6;

      // Kernel execution time
      clGetEventProfilingInfo(atomic_event_kernel, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, NULL);
      clGetEventProfilingInfo(atomic_event_kernel, CL_PROFILING_COMMAND_END,
                              sizeof(end), &end, NULL);
      atomic_time_kernel += (double)(end - start) * 1e-6;
      clGetEventProfilingInfo(non_atomic_event_kernel, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, NULL);
      clGetEventProfilingInfo(non_atomic_event_kernel, CL_PROFILING_COMMAND_END,
                              sizeof(end), &end, NULL);
      non_atomic_time_kernel += (double)(end - start) * 1e-6;

      // Read result buffer
      clGetEventProfilingInfo(atomic_event_read, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, NULL);
      clGetEventProfilingInfo(atomic_event_read, CL_PROFILING_COMMAND_END, sizeof(end),
                              &end, NULL);
      atomic_time_read += (double)(end - start) * 1e-6;
      clGetEventProfilingInfo(non_atomic_event_read, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, NULL);
      clGetEventProfilingInfo(non_atomic_event_read, CL_PROFILING_COMMAND_END,
                              sizeof(end), &end, NULL);
      non_atomic_time_read += (double)(end - start) * 1e-6;

      printf("\nConfiguration: L = %zu, B = %zu\n", L, B);
      printf("--- PROFILIRANJE (ATOMIC) ---\n");
      printf("Prijenos RAM -> GPU: %f ms\n", time_write);
      printf("Izvodenje (Kernel):  %f ms\n", atomic_time_kernel);
      printf("Prijenos GPU -> RAM: %f ms\n", atomic_time_read);
      printf("Ukupno (bez setupa): %f ms\n",
             time_write + atomic_time_kernel + atomic_time_read);
      printf("\n--- PROFILIRANJE (NON-ATOMIC) ---\n");
      printf("Prijenos RAM -> GPU: %f ms\n", time_write);
      printf("Izvodenje (Kernel):  %f ms\n", non_atomic_time_kernel);
      printf("Prijenos GPU -> RAM: %f ms\n", non_atomic_time_read);
      printf("Ukupno (bez setupa): %f ms\n",
             time_write + non_atomic_time_kernel + non_atomic_time_read);
    }
  }

  int cpu_count = 0;
  for (size_t i = 0; i < N; ++i) {
    if (is_prime[input[i]]) {
      ++cpu_count;
    }
  }
  if (cpu_count != atomic_result) {
    fprintf(stderr, "Atomic kernel result mismatch! CPU: %d, GPU: %d\n", cpu_count,
            atomic_result);
  } else {
    printf("Atomic kernel result matches CPU count: %d\n", atomic_result);
  }
  if (cpu_count != non_atomic_result) {
    fprintf(stderr, "Non-atomic kernel result mismatch! CPU: %d, GPU: %d\n", cpu_count,
            non_atomic_result);
  } else {
    printf("Non-atomic kernel result matches CPU count: %d\n", non_atomic_result);
  }

  // Cleanup
  CL_CHECK(clReleaseMemObject(input_buffer));
  CL_CHECK(clReleaseMemObject(result_buffer));

  // Cleanup
  CL_CHECK(clReleaseKernel(atomic_kernel));
  CL_CHECK(clReleaseKernel(non_atomic_kernel));
  CL_CHECK(clReleaseProgram(program));
  CL_CHECK(clReleaseContext(context));
  CL_CHECK(clReleaseCommandQueue(queue));

  return 0;
}
