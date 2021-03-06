#ifndef CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
#define CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h

#include <hip/hip_runtime.h>

#include "DataFormats/BeamSpotPOD.h"
#include "CUDACore/device_unique_ptr.h"

class BeamSpotCUDA {
public:
  // default constructor, required by cms::hip::Product<BeamSpotCUDA>
  BeamSpotCUDA() = default;

  // constructor that allocates cached device memory on the given CUDA stream
  BeamSpotCUDA(hipStream_t stream) { data_d_ = cms::hip::make_device_unique<BeamSpotPOD>(stream); }

  // movable, non-copiable
  BeamSpotCUDA(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA(BeamSpotCUDA&&) = default;
  BeamSpotCUDA& operator=(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA& operator=(BeamSpotCUDA&&) = default;

  BeamSpotPOD* data() { return data_d_.get(); }
  BeamSpotPOD const* data() const { return data_d_.get(); }

  cms::hip::device::unique_ptr<BeamSpotPOD>& ptr() { return data_d_; }
  cms::hip::device::unique_ptr<BeamSpotPOD> const& ptr() const { return data_d_; }

private:
  cms::hip::device::unique_ptr<BeamSpotPOD> data_d_;
};

#endif  // CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
