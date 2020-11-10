#include <cuda.h>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "CUDACore/cuteCheck.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  cuteCheck(cuMemAllocHost((void**)&gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU)));
  *gainForHLTonHost_ = gain;
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  cuteCheck(cuMemFreeHost((void*)gainForHLTonHost_));
}

SiPixelGainCalibrationForHLTGPU::GPUData::~GPUData() {
  cuteCheck(cuMemFree((CUdeviceptr)gainForHLTonGPU));
  cuteCheck(cuMemFree((CUdeviceptr)gainDataOnGPU));
}

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(CUstream stream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, CUstream stream) {
    cuteCheck(cuMemAlloc((CUdeviceptr*)&data.gainForHLTonGPU, sizeof(SiPixelGainForHLTonGPU)));
    cuteCheck(cuMemAlloc((CUdeviceptr*)&data.gainDataOnGPU, this->gainData_.size()));
    // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
    cuteCheck(cuMemcpyAsync(
        (CUdeviceptr)data.gainDataOnGPU, (CUdeviceptr)this->gainData_.data(), this->gainData_.size(), stream));
    cuteCheck(cuMemcpyAsync((CUdeviceptr)data.gainForHLTonGPU,
                            (CUdeviceptr)this->gainForHLTonHost_,
                            sizeof(SiPixelGainForHLTonGPU),
                            stream));
    cuteCheck(cuMemcpyAsync((CUdeviceptr)&data.gainForHLTonGPU->v_pedestals,
                            (CUdeviceptr)&data.gainDataOnGPU,
                            sizeof(SiPixelGainForHLTonGPU_DecodingStructure*),
                            stream));
  });
  return data.gainForHLTonGPU;
}
