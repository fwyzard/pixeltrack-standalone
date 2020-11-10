#ifndef HeterogeneousCore_CUDAUtilities_memsetAsync_h
#define HeterogeneousCore_CUDAUtilities_memsetAsync_h

#include "CUDACore/cuteCheck.h"
#include "CUDACore/device_unique_ptr.h"

#include <type_traits>

namespace cms {
  namespace cuda {
    template <typename T>
    inline void memsetAsync(device::unique_ptr<T>& ptr, T value, CUstream stream) {
      // Shouldn't compile for array types because of sizeof(T), but
      // let's add an assert with a more helpful message
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
      cuteCheck(cuMemsetD8Async((CUdeviceptr) ptr.get(), value, sizeof(T), stream));
    }

    template <typename T>
    inline void memsetAsync(device::unique_ptr<T[]>& ptr, unsigned char value, size_t nelements, CUstream stream) {
      cuteCheck(cuMemsetD8Async((CUdeviceptr) ptr.get(), value, nelements * sizeof(T), stream));
    }
  }  // namespace cuda
}  // namespace cms

#endif
