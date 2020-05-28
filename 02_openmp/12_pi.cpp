#include <cstdio>

int main() {
  int n = 10;
  double dx = 1. / n;
  double pi = 0;
<<<<<<< HEAD
#pragama omp parallel for reduction(+:pi)
=======
#pragma omp parallel for reduction(+:pi)
>>>>>>> ebc183ec04fbd943b6eda12f4665b266c3a700da
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  printf("%17.15f\n",pi);
}
