//  author: Felice Pantaleo, CERN, 2018
#include <cassert>
#include <iostream>
#include <new>

#include <cuda.h>
#include <cuda.h>

#include "CUDACore/SimpleVector.h"
#include "CUDACore/cuteCheck.h"
#include "CUDACore/requireDevices.h"

__global__ void vector_pushback(cms::cuda::SimpleVector<int> *foo) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  foo->push_back(index);
}

__global__ void vector_reset(cms::cuda::SimpleVector<int> *foo) { foo->reset(); }

__global__ void vector_emplace_back(cms::cuda::SimpleVector<int> *foo) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  foo->emplace_back(index);
}

int main() {
  cms::cudatest::requireDevices();

  auto maxN = 10000;
  cms::cuda::SimpleVector<int> *obj_ptr = nullptr;
  cms::cuda::SimpleVector<int> *d_obj_ptr = nullptr;
  cms::cuda::SimpleVector<int> *tmp_obj_ptr = nullptr;
  int *data_ptr = nullptr;
  int *d_data_ptr = nullptr;

  cuteCheck(cuMemAllocHost((void **)&obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  cuteCheck(cuMemAllocHost((void **)&data_ptr, maxN * sizeof(int)));
  cuteCheck(cuMemAlloc((CUdeviceptr *)&d_data_ptr, maxN * sizeof(int)));

  auto v = cms::cuda::make_SimpleVector(obj_ptr, maxN, data_ptr);

  cuteCheck(cuMemAllocHost((void **)&tmp_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  cms::cuda::make_SimpleVector(tmp_obj_ptr, maxN, d_data_ptr);
  assert(tmp_obj_ptr->size() == 0);
  assert(tmp_obj_ptr->capacity() == static_cast<int>(maxN));

  cuteCheck(cuMemAlloc((CUdeviceptr *)&d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));
  // ... and copy the object to the device.
  cuteCheck(cuMemcpy((CUdeviceptr)d_obj_ptr, (CUdeviceptr)tmp_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));

  int numBlocks = 5;
  int numThreadsPerBlock = 256;
  vector_pushback<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);
  cuteCheck(cuCtxSynchronize());

  cuteCheck(cuMemcpy((CUdeviceptr)obj_ptr, (CUdeviceptr)d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));
  vector_reset<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);
  cuteCheck(cuCtxSynchronize());

  cuteCheck(cuMemcpy((CUdeviceptr)obj_ptr, (CUdeviceptr)d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));

  assert(obj_ptr->size() == 0);

  vector_emplace_back<<<numBlocks, numThreadsPerBlock>>>(d_obj_ptr);
  cuteCheck(cuCtxSynchronize());

  cuteCheck(cuMemcpy((CUdeviceptr)obj_ptr, (CUdeviceptr)d_obj_ptr, sizeof(cms::cuda::SimpleVector<int>)));

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));

  cuteCheck(cuMemcpy((CUdeviceptr)data_ptr, (CUdeviceptr)d_data_ptr, obj_ptr->size() * sizeof(int)));
  cuteCheck(cuMemFreeHost((void *)obj_ptr));
  cuteCheck(cuMemFreeHost((void *)data_ptr));
  cuteCheck(cuMemFreeHost((void *)tmp_obj_ptr));
  cuteCheck(cuMemFree((CUdeviceptr)d_data_ptr));
  cuteCheck(cuMemFree((CUdeviceptr)d_obj_ptr));
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
