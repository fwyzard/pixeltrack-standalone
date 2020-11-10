#ifndef HeterogeneousCore_CUDAUtilities_eventWorkHasCompleted_h
#define HeterogeneousCore_CUDAUtilities_eventWorkHasCompleted_h

#include <cuda.h>

#include "CUDACore/cuteCheck.h"

namespace cms {
  namespace cuda {
    /**
   * Returns true if the work captured by the event (=queued to the
   * CUDA stream at the point of cuEventRecord()) has completed.
   *
   * Returns false if any captured work is incomplete.
   *
   * In case of errors, throws an exception.
   */
    inline bool eventWorkHasCompleted(CUevent event) {
      const auto ret = cuEventQuery(event);
      if (ret == CUDA_SUCCESS) {
        return true;
      } else if (ret == CUDA_ERROR_NOT_READY) {
        return false;
      }
      // leave error case handling to cuteCheck
      cuteCheck(ret);
      return false;  // to keep compiler happy
    }
  }  // namespace cuda
}  // namespace cms

#endif
