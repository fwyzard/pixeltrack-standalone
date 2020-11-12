#include "CUDACore/StreamCache.h"
#include "CUDACore/cuteCheck.h"
#include "CUDACore/currentDevice.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"

namespace cms::cuda {
  void StreamCache::Deleter::operator()(CUstream stream) const {
    if (device_ != -1) {
      ScopedSetDevice deviceGuard{device_};
      cuteCheck(cuStreamDestroy(stream));
    }
  }

  // StreamCache should be constructed by the first call to
  // getStreamCache() only if we have CUDA devices present
  StreamCache::StreamCache() : cache_(deviceCount()) {}

  SharedStreamPtr StreamCache::get() {
    const auto dev = currentDevice();
    return cache_[dev].makeOrGet([dev]() {
      CUstream stream;
      cuteCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
      return std::unique_ptr<BareStream, Deleter>(stream, Deleter{dev});
    });
  }

  void StreamCache::clear() {
    // Reset the contents of the caches, but leave an
    // edm::ReusableObjectHolder alive for each device. This is needed
    // mostly for the unit tests, where the function-static
    // StreamCache lives through multiple tests (and go through
    // multiple shutdowns of the framework).
    cache_.clear();
    cache_.resize(deviceCount());
  }

  StreamCache& getStreamCache() {
    // the public interface is thread safe
    static StreamCache cache;
    return cache;
  }
}  // namespace cms::cuda
