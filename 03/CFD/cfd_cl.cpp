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

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "arraymalloc.h"
#include "boundary.h"
#include "cfdio.h"
#include "jacobi.h"
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
  static constexpr int printfreq = 1000;  // output frequency
  float tolerance = 0.0;  // tolerance for convergence. <=0 means do not check

  // command line arguments
  int scalefactor, numiter;

  // simulation sizes
  static constexpr int bbase = 10;
  static constexpr int hbase = 15;
  static constexpr int wbase = 5;
  static constexpr int mbase = 32;
  static constexpr int nbase = 32;

  int checkerr = 0;

  // do we stop because of tolerance?
  if (tolerance > 0) {
    checkerr = 1;
  }

  // check command line parameters and parse them
  if (argc < 3 || argc > 4) {
    printf("Usage: cfd <scale> <numiter>\n");
    return 0;
  }

  scalefactor = atoi(argv[1]);
  numiter = atoi(argv[2]);

  if (!checkerr) {
    printf("Scale Factor = %i, iterations = %i\n", scalefactor, numiter);
  } else {
    printf("Scale Factor = %i, iterations = %i, tolerance= %g\n", scalefactor, numiter,
           tolerance);
  }

  printf("Irrotational flow\n");

  // Calculate b, h & w and m & n
  static const int b = bbase * scalefactor;
  static const int h = hbase * scalefactor;
  static const int w = wbase * scalefactor;
  static const int m = mbase * scalefactor;
  static const int n = nbase * scalefactor;

  printf("Running CFD on %d x %d grid in parallel\n", m, n);

  // main arrays
  size_t total_size = (m + 2) * (n + 2);
  std::vector<float> psi(total_size, 0.0);
  // temporary versions of main arrays
  std::vector<float> psitmp(total_size, 0.0);

  // set the psi boundary conditions
  boundarypsi(psi.data(), m, n, b, h, w);

  // compute normalisation factor for error
  float bnorm = 0.0;

  for (size_t i = 0; i < m + 2; ++i) {
    for (size_t j = 0; j < n + 2; ++j) {
      bnorm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j];
    }
  }
  bnorm = sqrt(bnorm);

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
  std::string file_name = "jacobi.cl";
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
  cl_kernel jacobi_kernel =
      (cl_kernel)CL_CHECK_ERR(clCreateKernel(program, "jacobi_kernel", &_err));

  // Create buffers and write to buffers
  cl_mem psi_buffer = (cl_mem)CL_CHECK_ERR(clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(float) * total_size, NULL, &_err));
  cl_mem psitmp_buffer = (cl_mem)CL_CHECK_ERR(clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(float) * total_size, NULL, &_err));
  CL_CHECK(clEnqueueWriteBuffer(queue, psi_buffer, CL_TRUE, 0,
                                sizeof(float) * total_size, psi.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(queue, psitmp_buffer, CL_TRUE, 0,
                                sizeof(float) * total_size, psi.data(), 0, NULL, NULL));

  // begin iterative Jacobi loop
  float error = -1;
  printf("\nStarting main loop...\n\n");

  const size_t global_work_size[2] = {static_cast<size_t>(m), static_cast<size_t>(n)};

  // Events for measurement
  cl_event event_read;
  cl_ulong start, end;
  float time_read = 0;

  struct timespec start_host, end_host;
  clock_gettime(CLOCK_MONOTONIC, &start_host);
  size_t iter = 1;
  for (; iter <= numiter; ++iter) {
    // Set kernel arguments
    //__kernel void jacobi_kernel(__global float *psinew, __global const float *psi,
    //                           const int m, const int n) {
    CL_CHECK(clSetKernelArg(jacobi_kernel, 0, sizeof(cl_mem), &psitmp_buffer));
    CL_CHECK(clSetKernelArg(jacobi_kernel, 1, sizeof(cl_mem), &psi_buffer));
    CL_CHECK(clSetKernelArg(jacobi_kernel, 2, sizeof(int), &m));
    CL_CHECK(clSetKernelArg(jacobi_kernel, 3, sizeof(int), &n));
    // Execute kernel and read result
    CL_CHECK(clEnqueueNDRangeKernel(queue, jacobi_kernel, 2, NULL, global_work_size,
                                    NULL, 0, NULL, NULL));

    // Swap buffers
    std::swap(psi_buffer, psitmp_buffer);
  }
  iter = std::min(iter, static_cast<size_t>(numiter));

  CL_CHECK(clFinish(queue));
  // Read back the buffers
  CL_CHECK(clEnqueueReadBuffer(queue, psi_buffer, CL_TRUE, 0,
                               sizeof(float) * total_size, psi.data(), 0, NULL,
                               &event_read));
  CL_CHECK(clEnqueueReadBuffer(queue, psitmp_buffer, CL_TRUE, 0,
                               sizeof(float) * total_size, psitmp.data(), 0, NULL,
                               &event_read));

  clock_gettime(CLOCK_MONOTONIC, &end_host);
  float time_kernel = (end_host.tv_sec - start_host.tv_sec) * 1e3 +
                      (end_host.tv_nsec - start_host.tv_nsec) * 1e-6;
  // Read result buffer
  clGetEventProfilingInfo(event_read, CL_PROFILING_COMMAND_START, sizeof(start), &start,
                          NULL);
  clGetEventProfilingInfo(event_read, CL_PROFILING_COMMAND_END, sizeof(end), &end,
                          NULL);
  time_read += (float)(end - start) * 1e-6;

  if (checkerr || iter == numiter) {
    error = deltasq(psitmp.data(), psi.data(), m, n);

    error = sqrt(error);
    error = error / bnorm;
  }

  float ttot = time_kernel + time_read;
  printf("--- PROFILIRANJE ---\n");
  printf("Izvodenje (Kernel):  %f ms\n", time_kernel);
  printf("Prijenos GPU -> RAM: %f ms\n", time_read);
  printf("Ukupno (bez setupa): %f ms\n", time_kernel + time_read);
  printf("After %ld iterations, the error is %g\n", iter, error);
  printf("Time for %ld iterations was %g ms\n", iter, ttot);
  printf("Each iteration took %g ms\n", ttot / iter);

  // Cleanup
  CL_CHECK(clReleaseMemObject(psi_buffer));
  CL_CHECK(clReleaseMemObject(psitmp_buffer));

  // Cleanup
  CL_CHECK(clReleaseKernel(jacobi_kernel));
  CL_CHECK(clReleaseProgram(program));
  CL_CHECK(clReleaseContext(context));
  CL_CHECK(clReleaseCommandQueue(queue));

  printf("... finished\n");

  return 0;
}
