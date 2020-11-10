#ifndef HeterogeneousCore_CUDAUtilities_allocate_device_h
#define HeterogeneousCore_CUDAUtilities_allocate_device_h

#include <cuda.h>

namespace cms {
  namespace cuda {
    // Allocate device memory
    void *allocate_device(int dev, size_t nbytes, CUstream stream);

    // Free device memory (to be called from unique_ptr)
    void free_device(int device, void *ptr);
  }  // namespace cuda
}  // namespace cms

#endif
