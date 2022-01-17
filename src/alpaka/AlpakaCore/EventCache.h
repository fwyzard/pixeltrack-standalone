#ifndef HeterogeneousCore_AlpakaUtilities_EventCache_h
#define HeterogeneousCore_AlpakaUtilities_EventCache_h

#include <memory>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/getDevIndex.h"
#include "Framework/ObjectCache.h"

namespace cms::alpakatools {

  template <typename Event>
  class EventCache {
  public:
    using Device = alpaka::Dev<Event>;
    using Platform = alpaka::Pltf<Device>;

    // EventCache should be constructed by the first call to getEventCache()
    EventCache() : cache_{std::make_unique<Cache[]>(alpaka::getDevCount<Platform>())} {}

    // Gets a (cached) alpaka event for the specified device.
    // The event will be returned to the cache by the shared_ptr destructor.
    // The returned event is guaranteed to be in the state where all
    // captured work has completed, i.e. alpaka::isComplete(...) == true.
    // This function is thread safe
    std::shared_ptr<Event> get(Device const& dev) {
      return cache_[cms::alpakatools::getDevIndex(dev)].get(std::in_place, dev);
    }

  private:
    // Not thread safe, intended to be called only from CUDAService destructor
    void clear() {
      // Reset the contents of the caches, but leave an
      // internal::ObjectCache alive for each device. This is needed
      // mostly for the unit tests, where the function-static
      // EventCache lives through multiple tests (and go through
      // multiple shutdowns of the framework).
      cache_ = std::make_unique<Cache[]>(alpaka::getDevCount<Platform>());
    }

    struct IsReady {
      bool operator()(Event const& e){ return alpaka::isComplete(e); }
    };
    using Cache = internal::ObjectCache<Event, IsReady>;

    std::unique_ptr<Cache[]> cache_;
  };

  // Gets the global instance of an EventCache
  // This function is thread safe
  template <typename Event>
  EventCache<Event>& getEventCache() {
    // the public interface is thread safe
    static EventCache<Event> cache;
    return cache;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaUtilities_EventCache_h
