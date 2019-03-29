#include "QPULib.h"

using namespace QPULib;

// Define function that runs on the GPU.

void hello(Ptr<Int> p)
{
  p[me() << 4] =  me();
}

int main()
{
  // Construct kernel
  auto k = compile(hello);
  k.setNumQPUs(12);
  // Allocate and initialise array shared between ARM and GPU
  SharedArray<int> array(16 * k.numQPUs);
  for (int i = 0; i < (16 * k.numQPUs); i++)
    array[i] = 100;

  // Invoke the kernel and display the result
  k(&array);
  for (int i = 0; i < (k.numQPUs * 16); i++) {
    printf("%i: %i\n", i, array[i]);
  }

  return 0;
}
