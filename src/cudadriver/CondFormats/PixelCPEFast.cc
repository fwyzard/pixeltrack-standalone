#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda.h>

#include "Geometry/phase1PixelTopology.h"
#include "CUDACore/cuteCheck.h"
#include "CondFormats/PixelCPEFast.h"

// Services
// this is needed to get errors from templates

namespace {
  constexpr float micronsToCm = 1.0e-4;
}

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(std::string const &path) {
  {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    in.read(reinterpret_cast<char *>(&m_commonParamsGPU), sizeof(pixelCPEforGPU::CommonParams));
    unsigned int ndetParams;
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    m_detParamsGPU.resize(ndetParams);
    in.read(reinterpret_cast<char *>(m_detParamsGPU.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    in.read(reinterpret_cast<char *>(&m_averageGeometry), sizeof(pixelCPEforGPU::AverageGeometry));
    in.read(reinterpret_cast<char *>(&m_layerGeometry), sizeof(pixelCPEforGPU::LayerGeometry));
  }

  cpuData_ = {
      &m_commonParamsGPU,
      m_detParamsGPU.data(),
      &m_layerGeometry,
      &m_averageGeometry,
  };
}

const pixelCPEforGPU::ParamsOnGPU *PixelCPEFast::getGPUProductAsync(CUstream stream) const {
  const auto &data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData &data, CUstream stream) {
    // and now copy to device...
    cuteCheck(cuMemAlloc((CUdeviceptr *)&data.h_paramsOnGPU.m_commonParams, sizeof(pixelCPEforGPU::CommonParams)));
    cuteCheck(cuMemAlloc((CUdeviceptr *)&data.h_paramsOnGPU.m_detParams,
                         this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams)));
    cuteCheck(
        cuMemAlloc((CUdeviceptr *)&data.h_paramsOnGPU.m_averageGeometry, sizeof(pixelCPEforGPU::AverageGeometry)));
    cuteCheck(cuMemAlloc((CUdeviceptr *)&data.h_paramsOnGPU.m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry)));
    cuteCheck(cuMemAlloc((CUdeviceptr *)&data.d_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU)));

    cuteCheck(cuMemcpyAsync(
        (CUdeviceptr)data.d_paramsOnGPU, (CUdeviceptr)&data.h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU), stream));
    cuteCheck(cuMemcpyAsync((CUdeviceptr)data.h_paramsOnGPU.m_commonParams,
                            (CUdeviceptr) & this->m_commonParamsGPU,
                            sizeof(pixelCPEforGPU::CommonParams),
                            stream));
    cuteCheck(cuMemcpyAsync((CUdeviceptr)data.h_paramsOnGPU.m_averageGeometry,
                            (CUdeviceptr) & this->m_averageGeometry,
                            sizeof(pixelCPEforGPU::AverageGeometry),
                            stream));
    cuteCheck(cuMemcpyAsync((CUdeviceptr)data.h_paramsOnGPU.m_layerGeometry,
                            (CUdeviceptr) & this->m_layerGeometry,
                            sizeof(pixelCPEforGPU::LayerGeometry),
                            stream));
    cuteCheck(cuMemcpyAsync((CUdeviceptr)data.h_paramsOnGPU.m_detParams,
                            (CUdeviceptr)this->m_detParamsGPU.data(),
                            this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams),
                            stream));
  });
  return data.d_paramsOnGPU;
}

PixelCPEFast::GPUData::~GPUData() {
  if (d_paramsOnGPU != nullptr) {
    cuMemFree((CUdeviceptr)h_paramsOnGPU.m_commonParams);
    cuMemFree((CUdeviceptr)h_paramsOnGPU.m_detParams);
    cuMemFree((CUdeviceptr)h_paramsOnGPU.m_averageGeometry);
    cuMemFree((CUdeviceptr)h_paramsOnGPU.m_layerGeometry);
    cuMemFree((CUdeviceptr)d_paramsOnGPU);
  }
}
