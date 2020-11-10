#ifndef HeterogeneousCore_CUDAUtilities_HostAllocator_h
#define HeterogeneousCore_CUDAUtilities_HostAllocator_h

#include <memory>
#include <new>
#include <cuda.h>

namespace cms {
  namespace cuda {

    class bad_alloc : public std::bad_alloc {
    public:
      bad_alloc(CUresult error) noexcept : error_(error) {}

      const char* what() const noexcept override {
        const char* msg;
        cuGetErrorString(error_, &msg);
        return msg;
      }

    private:
      CUresult error_;
    };

    template <typename T, unsigned int FLAGS = 0>
    class HostAllocator {
    public:
      using value_type = T;

      template <typename U>
      struct rebind {
        using other = HostAllocator<U, FLAGS>;
      };

      T* allocate(std::size_t n) const __attribute__((warn_unused_result)) __attribute__((malloc))
      __attribute__((returns_nonnull)) {
        void* ptr = nullptr;
        CUresult status = cuMemHostAlloc(&ptr, n * sizeof(T), FLAGS);
        if (status != CUDA_SUCCESS) {
          throw bad_alloc(status);
        }
        if (ptr == nullptr) {
          throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
      }

      void deallocate(T* p, std::size_t n) const {
        CUresult status = cuMemFreeHost(p);
        if (status != CUDA_SUCCESS) {
          throw bad_alloc(status);
        }
      }
    };

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_HostAllocator_h
