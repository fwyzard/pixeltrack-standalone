#ifndef HeterogenousCore_CUDAUtilities_deviceCount_h
#define HeterogenousCore_CUDAUtilities_deviceCount_h

#include "CUDACore/cuteCheck.h"

#include <cuda.h>

namespace cms {
  namespace cuda {
    inline int deviceCount() {
      int ndevices;
      cuteCheck(cuDeviceGetCount(&ndevices));
      return ndevices;
    }
  }  // namespace cuda
}  // namespace cms

#endif
