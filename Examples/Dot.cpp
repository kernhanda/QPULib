#include <sys/time.h>
#include <math.h>
#include <QPULib.h>

#include <algorithm>
#include <numeric>
#include <random>

#include <openblas/cblas.h>

using namespace QPULib;

void dot(Int N, Ptr<Float> A, Ptr<Float> B, Ptr<Float> result)
{
    Int inc = 16;//numQPUs() << 4;
    Int qpuID = me();
    Ptr<Float> a = A + index() + (qpuID << 4);
    Ptr<Float> b = B + index() + (qpuID << 4);
    Ptr<Float> c = result + index() + (qpuID << 4);
    gather(a); gather(b);

    Float aOld, bOld;
    Float temp = result[qpuID << 4];
    For (Int i = 0, i < N, i = i + inc)
        gather(a + i + inc+ (qpuID << 4));  gather(b + i + inc+ (qpuID << 4));
        receive(aOld); receive(bOld);
        temp = temp + (aOld * bOld);
        // store(aOld * bOld, c + i+ (qpuID << 4));
    End

    store(temp, c);
}

// ============================================================================
// Main
// ============================================================================

int main()
{
  // Timestamps
  timeval tvStart, tvEnd, tvDiff;

  const int N = 16 * 24; // 192000

  // Construct kernel
  auto k = compile(dot);

  // Use 12 QPUs
  k.setNumQPUs(12);

  // Allocate and initialise arrays shared between ARM and GPU
  SharedArray<float> x(N), y(N), result(16 * k.numQPUs);
  std::fill_n(&result[0], result.size, 0.0f);
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::uniform_real_distribution<float> dist;
  std::generate_n(&x[0], N, [&] { return dist(engine); });
  std::generate_n(&y[0], N, [&] { return dist(engine); });

  gettimeofday(&tvStart, NULL);
  k(N, &x, &y, &result);
  float gpuOut = std::accumulate(&result[0], &result[0] + result.size, 0.f);
  gettimeofday(&tvEnd, NULL);
  timersub(&tvEnd, &tvStart, &tvDiff);
  printf("GPU: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);
  printf("gpuOutput = %f\n", gpuOut);

  gettimeofday(&tvStart, NULL);
  float blasOut = cblas_sdot(N, &x[0], 1, &y[0], 1);
  gettimeofday(&tvEnd, NULL);
  timersub(&tvEnd, &tvStart, &tvDiff);
  printf("BLAS: %ld.%06lds\n", tvDiff.tv_sec, tvDiff.tv_usec);
  printf("blasOutput = %f\n", blasOut);

  // printf("\nIndex,A,B,GPU,CPU,Delta\n");
  // for (int i = 0; i < N; ++i) {
  //   float a = x[i], b = y[i];
  //   float gpu = result[i];
  //   float cpu = a * b;
  //   float delta = cpu - gpu;
  //   printf("%d,%f,%f,%f,%f,%f\n", i, a, b, gpu, cpu, delta);
  // }

  return 0;
}
