#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <assert.h>
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif
#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)
#define checkLastCudaError()\
  do { \
    cudaError_t err{cudaGetLastError()};\
    if (err != cudaSuccess)\
    {\
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
        exit(EXIT_FAILURE); \
    }\
  } while (0)\

#include <eigen3/Eigen/Dense>
#include "models.h"
#include "cuda.h"
#include "lu_decomposition.cuh"
#include "cusolverDn.h"

void createInterpolationGPU(const std::vector<DataPoint>* observedData, TheoreticalParam param, double** krigingOutput, size_t maxX, size_t maxY, size_t size);