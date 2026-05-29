#include "jacobi.h"

void jacobistep(double *psinew, double *psi, int m, int n) {
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      psinew[i * (m + 2) + j] =
          0.25 * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] +
                  psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1]);
    }
  }
}

double deltasq(double *newarr, double *oldarr, int m, int n) {
  double dsq = 0.0;
  double tmp;

  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
      dsq += tmp * tmp;
    }
  }

  return dsq;
}
