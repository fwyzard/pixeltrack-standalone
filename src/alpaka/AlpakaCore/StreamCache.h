#ifndef HeterogeneousCore_AlpakaUtilities_StreamCache_h
#define HeterogeneousCore_AlpakaUtilities_StreamCache_h

#include <memory>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/getDevIndex.h"
#include "Framework/ObjectCache.h"

namespace cms::alpakatools {

  template <typename Queue>
  class StreamCache {
  public:
    using Device = alpaka::Dev<Queue>;
    using Platform = alpaka::Pltf<Device>;

    // StreamCache should be constructed by the first call to
    // getStreamCache() only if we have CUDA devices present
    StreamCache() : cache_{std::make_unique<Cache[]>(alpaka::getDevCount<Platform>())} {}

    // Gets a (cached) CUDA stream for the current device. The stream
    // will be returned to the cache by the shared_ptr destructor.
    // This function is thread safe
    std::shared_ptr<Queue> get(Device const& dev) {
      return cache_[cms::alpakatools::getDevIndex(dev)].get(std::in_place, dev);
    }

  private:
    // Not thread safe, intended to be called only from CUDAService destructor
    void clear() {
      // Reset the contents of the caches, but leave an
      // internal::ObjectCache alive for each device. This is needed
      // mostly for the unit tests, where the function-static
      // StreamCache lives through multiple tests (and go through
      // multiple shutdowns of the framework).
      cache_ = std::make_unique<Cache[]>(alpaka::getDevCount<Platform>());
    }

    struct IsReady {
      bool operator()(Queue const& queue){ return alpaka::empty(queue); }
    };
    using Cache = internal::ObjectCache<Queue, IsReady>;

    std::unique_ptr<Cache[]> cache_;
  };

  // Gets the global instance of a StreamCache
  // This function is thread safe
  template <typename Queue>
  StreamCache<Queue>& getStreamCache() {
    // the public interface is thread safe
    static StreamCache<Queue> cache;
    return cache;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaUtilities_StreamCache_h
