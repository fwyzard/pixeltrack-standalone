#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cuteCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(CUstream stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(CUstream stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(2001, stream);
  cuteCheck(cuMemcpyAsync((CUdeviceptr)ret.get(), (CUdeviceptr)m_hitsModuleStart, 4 * 2001, stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::globalCoordToHostAsync(CUstream stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cuteCheck(cuMemcpyAsync(
      (CUdeviceptr)ret.get(), (CUdeviceptr)m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<int32_t[]> TrackingRecHit2DCUDA::chargeToHostAsync(CUstream stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nHits(), stream);
  cuteCheck(cuMemcpyAsync(
      (CUdeviceptr)ret.get(), (CUdeviceptr)m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(CUstream stream) const {
  auto ret = cms::cuda::make_host_unique<int16_t[]>(2 * nHits(), stream);
  cuteCheck(cuMemcpyAsync(
      (CUdeviceptr)ret.get(), (CUdeviceptr)m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), stream));
  return ret;
}
