#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <limits>

#include "SYCLCore/ScopedSetDevice.h"
#include "SYCLCore/allocate_device.h"
#include "SYCLCore/cudaCheck.h"

#include "getCachingDeviceAllocator.h"

namespace {
  const size_t maxAllocationSize =
      notcub::CachingDeviceAllocator::IntPow(cms::cuda::allocator::binGrowth, cms::cuda::allocator::maxBin);
}

namespace cms::cuda {
  void *allocate_device(int dev, size_t nbytes, sycl::queue *stream) {
    void *ptr = nullptr;
    if constexpr (allocator::useCaching) {
      if (nbytes > maxAllocationSize) {
        throw std::runtime_error("Tried to allocate " + std::to_string(nbytes) +
                                 " bytes, but the allocator maximum is " + std::to_string(maxAllocationSize));
      }
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceAllocate(dev, &ptr, nbytes, stream));
    } else {
      ScopedSetDevice setDeviceForThisScope(dev);
      cudaCheck((ptr = (void *)sycl::malloc_device(nbytes, dpct::get_default_queue()), 0));
    }
    return ptr;
  }

  void free_device(int device, void *ptr) {
    if constexpr (allocator::useCaching) {
      cudaCheck(allocator::getCachingDeviceAllocator().DeviceFree(device, ptr));
    } else {
      ScopedSetDevice setDeviceForThisScope(device);
      cudaCheck((sycl::free(ptr, dpct::get_default_queue()), 0));
    }
  }

}  // namespace cms::cuda
