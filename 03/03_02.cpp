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

int main(int argc, char **argv) {
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
  std::string file_name = "pi.cl";
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

  // Create kernel
  cl_kernel pi_kernel =
      (cl_kernel)CL_CHECK_ERR(clCreateKernel(program, "partial_pi", &_err));
  cl_kernel sum_reduce_kernel =
      (cl_kernel)CL_CHECK_ERR(clCreateKernel(program, "sum_reduction", &_err));

  // Events for measurement
  cl_event event_read;
  cl_event event_kernel;

  // Config
  const size_t L = 256;  // Threads per block
  const size_t B = 256;  // Number of blocks
  const size_t G = B * L;
  const size_t global_work_size[1] = {G};
  const size_t local_work_size[1] = {L};
  const float zero = 0;
  const size_t N = (1 << 24);  // 2^20
  std::vector<float> results(B);
  float result = 0;

  // Create buffers
  // Results buffer (each thread gets its own memory spot)
  cl_mem result_buffer = (cl_mem)CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &_err));
  cl_mem scan_buffer = (cl_mem)CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * N, NULL, &_err));
  cl_mem block_sums_buffer = (cl_mem)CL_CHECK_ERR(
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * B, NULL, &_err));

  // Set kernel arguments
  CL_CHECK(clSetKernelArg(pi_kernel, 0, sizeof(int), &N));
  CL_CHECK(clSetKernelArg(pi_kernel, 1, sizeof(cl_mem), &result_buffer));

  //__kernel void sum_reduction(__global const float *input, __global float *output,
  //                            const int input_size, __local float *reduction_sums);
  CL_CHECK(clSetKernelArg(sum_reduce_kernel, 0, sizeof(cl_mem), &result_buffer));
  CL_CHECK(clSetKernelArg(sum_reduce_kernel, 1, sizeof(cl_mem), &block_sums_buffer));
  CL_CHECK(clSetKernelArg(sum_reduce_kernel, 2, sizeof(int), &N));
  CL_CHECK(clSetKernelArg(sum_reduce_kernel, 3, sizeof(float) * L, NULL));

  // Execute kernel and read result
  CL_CHECK(clEnqueueWriteBuffer(queue, result_buffer, CL_TRUE, 0, sizeof(float), &zero,
                                0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(queue, pi_kernel, 1, NULL, global_work_size,
                                  local_work_size, 0, NULL, &event_kernel));
  CL_CHECK(clEnqueueNDRangeKernel(queue, sum_reduce_kernel, 1, NULL, global_work_size,
                                  local_work_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(queue, block_sums_buffer, CL_TRUE, 0, sizeof(float) * B,
                               results.data(), 0, NULL, &event_read));

  for (size_t i = 0; i < B; ++i) {
    result += results[i];
  }

  // Asynchronous finish
  CL_CHECK(clFinish(queue));

  cl_ulong start, end;
  double time_kernel = 0;
  double time_read = 0;

  // Kernel execution time
  clGetEventProfilingInfo(event_kernel, CL_PROFILING_COMMAND_START, sizeof(start),
                          &start, NULL);
  clGetEventProfilingInfo(event_kernel, CL_PROFILING_COMMAND_END, sizeof(end), &end,
                          NULL);
  time_kernel += (double)(end - start) * 1e-6;

  // Read result buffer
  clGetEventProfilingInfo(event_read, CL_PROFILING_COMMAND_START, sizeof(start), &start,
                          NULL);
  clGetEventProfilingInfo(event_read, CL_PROFILING_COMMAND_END, sizeof(end), &end,
                          NULL);
  time_read += (double)(end - start) * 1e-6;

  printf("Configuration: L = %zu, B = %zu\n", L, B);
  printf("--- PROFILIRANJE ---\n");
  printf("Izvodenje (Kernel):  %f ms\n", time_kernel);
  printf("Prijenos GPU -> RAM: %f ms\n", time_read);
  printf("Ukupno (bez setupa): %f ms\n", time_kernel + time_read);
  printf("Pi approximation: %.10f\n", result * 4 / N);

  double res = 0.0F;
  clock_t start_cpu = clock();
  for (size_t i = 0; i < N; ++i) {
    double current = ((i - 0.5) / N) * ((i - 0.5) / N);
    res += 1.0 / (1.0 + current);
  }

  res *= 4.0F / N;
  clock_t end_cpu = clock();
  double cpu_time = (double)(end_cpu - start_cpu) * 1e3 / CLOCKS_PER_SEC;
  printf("CPU Pi approximation: %.10f\n", res);
  printf("CPU time: %f ms\n", cpu_time);

  // Cleanup
  CL_CHECK(clReleaseMemObject(result_buffer));
  CL_CHECK(clReleaseMemObject(scan_buffer));
  CL_CHECK(clReleaseMemObject(block_sums_buffer));

  // Cleanup
  CL_CHECK(clReleaseKernel(pi_kernel));
  CL_CHECK(clReleaseProgram(program));
  CL_CHECK(clReleaseContext(context));
  CL_CHECK(clReleaseCommandQueue(queue));

  return 0;
}
