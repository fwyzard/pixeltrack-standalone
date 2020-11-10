#ifndef HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h
#define HeterogeneousCore_CUDAUtilities_ScopedSetDevice_h

#include "CUDACore/cuteCheck.h"

#include <cuda.h>

namespace cms {
  namespace cuda {
    class ScopedSetDevice {
    public:
      explicit ScopedSetDevice(int newDevice) {
        cuteCheck(cudaGetDevice(&prevDevice_));
        cuteCheck(cudaSetDevice(newDevice));
      }

      ~ScopedSetDevice() {
        // Intentionally don't check the return value to avoid
        // exceptions to be thrown. If this call fails, the process is
        // doomed anyway.
        cudaSetDevice(prevDevice_);
      }

    private:
      int prevDevice_;
    };
  }  // namespace cuda
}  // namespace cms

#endif
