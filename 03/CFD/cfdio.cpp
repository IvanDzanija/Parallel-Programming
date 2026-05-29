#include "cfdio.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "arraymalloc.h"

#pragma warning(disable : 4996)

// time in seconds
#ifdef _WIN32
#include <windows.h>
float gettime(void) {
  LARGE_INTEGER fr, t;
  QueryPerformanceFrequency(&fr);
  QueryPerformanceCounter(&t);
  float t_sec = (t.QuadPart) / (float)fr.QuadPart;
  return t_sec;
}
#else
#include <sys/time.h>
float gettime(void) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec + tp.tv_usec / (float)1.0e6;
}
#endif
