#ifndef HeterogeneousCore_SYCLUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_SYCLUtilities_interface_host_unique_ptr_h

#include <functional>
#include <memory>
#include <optional>

#include <CL/sycl.hpp>

namespace cms {
  namespace sycltools {
    namespace host {
      namespace impl {
        // Additional layer of types to distinguish from host::unique_ptr
        class HostDeleter {
        public:
          HostDeleter() = default;  // for edm::Wrapper
          HostDeleter(sycl::queue stream) : stream_{stream} {}

          void operator()(void *ptr) {
            if (stream_) {
              sycl::free(ptr, *stream_);
            }
          }

        private:
          std::optional<sycl::queue> stream_;
        };
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::HostDeleter>;

      namespace impl {
        template <typename T>
        struct make_host_unique_selector {
          using non_array = cms::sycltools::host::unique_ptr<T>;
        };
        template <typename T>
        struct make_host_unique_selector<T[]> {
          using unbounded_array = cms::sycltools::host::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_host_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace host

    // Allocate pinned host memory
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique(sycl::queue stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      void *mem = sycl::malloc_host(sizeof(T), stream);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                          host::impl::HostDeleter{stream}};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique(size_t n, sycl::queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      void *mem = sycl::malloc_host(n * sizeof(element_type), stream);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::HostDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename host::impl::make_host_unique_selector<T>::bounded_array make_host_unique(Args &&...) = delete;

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique_uninitialized(sycl::queue stream) {
      void *mem = sycl::malloc_host(sizeof(T), stream);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                          host::impl::HostDeleter{stream}};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique_uninitialized(
        size_t n, sycl::queue stream) {
      using element_type = typename std::remove_extent<T>::type;
      void *mem = sycl::malloc_host(n * sizeof(element_type), stream);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::HostDeleter{stream}};
    }

    template <typename T, typename... Args>
    typename host::impl::make_host_unique_selector<T>::bounded_array make_host_unique_uninitialized(Args &&...) = delete;
  }  // namespace sycltools
}  // namespace cms

#endif
