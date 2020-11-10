// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda.h>

// CMSSW includes
#include "CUDACore/cuteCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  cuteCheck(cuMemAllocHost((void**)&cablingMapHost, sizeof(SiPixelFedCablingMapGPU)));
  std::memcpy(cablingMapHost, &cablingMap, sizeof(SiPixelFedCablingMapGPU));

  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() { cuteCheck(cuMemFreeHost((void*)cablingMapHost)); }

const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(CUstream stream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, CUstream stream) {
    // allocate
    cuteCheck(cuMemAlloc((CUdeviceptr*)&data.cablingMapDevice, sizeof(SiPixelFedCablingMapGPU)));

    // transfer
    cuteCheck(cuMemcpyAsync(
        (CUdeviceptr)data.cablingMapDevice, (CUdeviceptr)this->cablingMapHost, sizeof(SiPixelFedCablingMapGPU), stream));
  });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(CUstream stream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(stream, [this](ModulesToUnpack& data, CUstream stream) {
        cuteCheck(cuMemAlloc((CUdeviceptr*)&data.modToUnpDefault, pixelgpudetails::MAX_SIZE_BYTE_BOOL));
        cuteCheck(cuMemcpyAsync((CUdeviceptr)data.modToUnpDefault,
                                (CUdeviceptr)this->modToUnpDefault.data(),
                                this->modToUnpDefault.size() * sizeof(unsigned char),
                                stream));
      });
  return data.modToUnpDefault;
}

SiPixelFedCablingMapGPUWrapper::GPUData::~GPUData() { cuteCheck(cuMemFree((CUdeviceptr)cablingMapDevice)); }

SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::~ModulesToUnpack() {
  cuteCheck(cuMemFree((CUdeviceptr)modToUnpDefault));
}
