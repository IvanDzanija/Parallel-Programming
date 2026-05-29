#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <vector>

#include "arraymalloc.h"
#include "boundary.h"
#include "cfdio.h"
#include "jacobi.h"

// WHY IS ACADEMIC CODE ALWAYS SO UGLY?
// DECLARING ALL VARIABLE AT THE TOP???? LOOP ITERATORS AT THE TOP OF THE CODE???

int main(int argc, char **argv) {
  static constexpr int printfreq = 1000;  // output frequency
  double tolerance = 0.0;  // tolerance for convergence. <=0 means do not check

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

  printf("Running CFD on %d x %d grid in serial\n", m, n);

  // main arrays
  std::vector<double> psi((m + 2) * (n + 2), 0.0);
  // temporary versions of main arrays
  std::vector<double> psitmp((m + 2) * (n + 2), 0.0);

  // set the psi boundary conditions
  boundarypsi(psi.data(), m, n, b, h, w);

  // compute normalisation factor for error
  double bnorm = 0.0;

  for (size_t i = 0; i < m + 2; ++i) {
    for (size_t j = 0; j < n + 2; ++j) {
      bnorm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j];
    }
  }
  bnorm = sqrt(bnorm);

  // begin iterative Jacobi loop
  double error;
  printf("\nStarting main loop...\n\n");
  double tstart = gettime();

  size_t iter = 1;
  for (; iter <= numiter; ++iter) {
    // calculate psi for next iteration
    jacobistep(psitmp.data(), psi.data(), m, n);

    // calculate current error if required
    if (checkerr || iter == numiter) {
      error = deltasq(psitmp.data(), psi.data(), m, n);

      error = sqrt(error);
      error = error / bnorm;
    }

    // quit early if we have reached required tolerance
    if (checkerr) {
      if (error < tolerance) {
        printf("Converged on iteration %ld\n", iter);
        break;
      }
    }

    // copy back
    for (size_t i = 1; i <= m; ++i) {
      for (size_t j = 1; j <= n; ++j) {
        psi[i * (m + 2) + j] = psitmp[i * (m + 2) + j];
      }
    }

    // print loop information
    if (iter % printfreq == 0) {
      if (!checkerr) {
        printf("Completed iteration %ld\n", iter);
      } else {
        printf("Completed iteration %ld, error = %g\n", iter, error);
      }
    }
  }  // iter

  if (iter > numiter) {
    iter = numiter;
  }

  double tstop = gettime();
  double ttot = tstop - tstart;

  // print out some stats
  printf("\n... finished\n");
  printf("After %ld iterations, the error is %g\n", iter, error);
  printf("Time for %ld iterations was %g seconds\n", iter, ttot);
  printf("Each iteration took %g seconds\n", ttot / iter);

  // output results
  // writedatafiles(psi,m,n, scalefactor);
  // writeplotfile(m,n,scalefactor);

  printf("... finished\n");

  return 0;
}
