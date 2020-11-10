#ifndef HeterogeneousCore_CUDAUtilities_SharedEventPtr_h
#define HeterogeneousCore_CUDAUtilities_SharedEventPtr_h

#include <memory>
#include <type_traits>

#include <cuda.h>

namespace cms {
  namespace cuda {
    // CUevent itself is a typedef for a pointer, for the use with
    // edm::ReusableObjectHolder the pointed-to type is more interesting
    // to avoid extra layer of indirection
    using SharedEventPtr = std::shared_ptr<std::remove_pointer_t<CUevent>>;
  }  // namespace cuda
}  // namespace cms

#endif
