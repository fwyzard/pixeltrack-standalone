#ifndef HeterogenousCore_CUDAUtilities_currentDevice_h
#define HeterogenousCore_CUDAUtilities_currentDevice_h

#include "CUDACore/cuteCheck.h"

#include <cuda.h>

namespace cms {
  namespace cuda {
    inline int currentDevice() {
      int dev;
      cuteCheck(cudaGetDevice(&dev));
      return dev;
    }
  }  // namespace cuda
}  // namespace cms

#endif
