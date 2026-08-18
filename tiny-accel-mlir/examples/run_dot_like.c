#include <stdio.h>
#include <stdlib.h>

/* Defined in LLVM IR compiled from tinyaccel (see compile_native.sh). */
float dot_like(float a, float b, float c);

int main(int argc, char **argv) {
  float a = argc > 1 ? strtof(argv[1], NULL) : 2.f;
  float b = argc > 2 ? strtof(argv[2], NULL) : 3.f;
  float c = argc > 3 ? strtof(argv[3], NULL) : 4.f;
  printf("dot_like(%g, %g, %g) = %g\n", a, b, c, dot_like(a, b, c));
  return 0;
}
