#include "jacobi.h"

void jacobistep(float *psinew, float *psi, int m, int n) {
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      psinew[i * (m + 2) + j] =
          0.25 * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] +
                  psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1]);
    }
  }
}

float deltasq(float *newarr, float *oldarr, int m, int n) {
  float dsq = 0.0;
  float tmp;

  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
      dsq += tmp * tmp;
    }
  }

  return dsq;
}
