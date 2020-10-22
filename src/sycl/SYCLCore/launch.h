#ifndef HeterogeneousCore_CUDAUtilities_launch_h
#define HeterogeneousCore_CUDAUtilities_launch_h

#include <tuple>

#include <CL/sycl.hpp>

#include "SYCLCore/cudaCheck.h"

/*
 * `cms::sycltools::launch` and `cms::sycltools::launch_cooperative` are wrappers around
 * the CUDA Runtime API calls to setup and call a CUDA kernel from the host.
 *
 * `kernel` should be a pointer to a __global__ void(...) function.
 * `config` describe the launch configuration: the grid size and block size, the
 *          dynamic shared memory size (default to 0) and the CUDA stream to use
 *          (default to 0, the default stream).
 * `args` are the arguments passed (by value) to the kernel.
 *
 *  Currently this is requires an extra copy to perform the necessary implicit
 *  conversions and ensure that the arguments match the kernel function signature;
 *  the extra copy could eventually be avoided for arguments that are already of
 *  the exact type.
 *
 *  Unlike the `kernel<<<...>>>(...)` syntax and the `cuda::launch(...)` 
 *  implementation from the CUDA API Wrappers, `cms::sycltools::launch(...)` and 
 *  `cms::sycltools::launch_cooperative` can be called from standard C++ host code.
 *
 *  Possible optimisations
 *
 *    - once C++17 is available in CUDA, replace the `pointer_setter` functor
 *      with a simpler function using fold expressions:
 *
 *  template<int N, class Tuple, std::size_t... Is>
 *  void pointer_setter(void* ptrs[N], Tuple const& t, std::index_sequence<Is...>)
 *  {
 *    ((ptrs[Is] = & std::get<Is>(t)), ...);
 *  }
 *
 *    - add a template specialisation to `launch` and `launch_cooperative` to
 *      avoid making a temporary copy of the parameters when they match the
 *      kernel signature.
 */

namespace cms {
  namespace sycltools {

    struct LaunchParameters {
      sycl::range<3> gridDim;
      sycl::range<3> blockDim;
      size_t sharedMem;
      sycl::queue* stream;

      LaunchParameters(sycl::range<3> gridDim,
                       sycl::range<3> blockDim,
                       size_t sharedMem = 0,
                       sycl::queue* stream = nullptr)
          : gridDim(gridDim), blockDim(blockDim), sharedMem(sharedMem), stream(stream) {}

      LaunchParameters(int gridDim, int blockDim, size_t sharedMem = 0, sycl::queue* stream = nullptr)
          : gridDim(gridDim), blockDim(blockDim), sharedMem(sharedMem), stream(stream) {}
    };

    namespace detail {

      template <typename T>
      struct kernel_traits;

      template <typename... Args>
      struct kernel_traits<void(Args...)> {
        static constexpr size_t arguments_size = sizeof...(Args);

        using argument_type_tuple = std::tuple<Args...>;

        template <size_t i>
        using argument_type = typename std::tuple_element<i, argument_type_tuple>::type;
      };

      // fill an array with the pointers to the elements of a tuple
      template <int I>
      struct pointer_setter {
        template <typename Tuple>
        void operator()(void const* ptrs[], Tuple const& t) {
          pointer_setter<I - 1>()(ptrs, t);
          ptrs[I - 1] = &std::get<I - 1>(t);
        }
      };

      template <>
      struct pointer_setter<0> {
        template <typename Tuple>
        void operator()(void const* ptrs[], Tuple const& t) {}
      };

    }  // namespace detail

    // wrappers for cudaLaunchKernel

    inline void launch(void (*kernel)(), LaunchParameters config) {
      /*
      DPCT1004:72: Could not generate replacement.
      */
      cudaCheck(cudaLaunchKernel(
          (const void*)kernel, config.gridDim, config.blockDim, nullptr, config.sharedMem, config.stream));
    }

    template <typename F, typename... Args>
#if __cplusplus >= 201703L
    std::enable_if_t<std::is_invocable_r<void, F, Args&&...>::value>
#else
    std::enable_if_t<std::is_void<std::result_of_t<F && (Args && ...)> >::value>
#endif
    launch(F* kernel, LaunchParameters config, Args&&... args) {
      using function_type = detail::kernel_traits<F>;
      typename function_type::argument_type_tuple args_copy(args...);

      constexpr auto size = function_type::arguments_size;
      void const* pointers[size];

      detail::pointer_setter<size>()(pointers, args_copy);
      /*
      DPCT1004:73: Could not generate replacement.
      */
      cudaCheck(cudaLaunchKernel(
          (const void*)kernel, config.gridDim, config.blockDim, (void**)pointers, config.sharedMem, config.stream));
    }

    // wrappers for cudaLaunchCooperativeKernel

    inline void launch_cooperative(void (*kernel)(), LaunchParameters config) {
      /*
      DPCT1004:74: Could not generate replacement.
      */
      cudaCheck(cudaLaunchCooperativeKernel(
          (const void*)kernel, config.gridDim, config.blockDim, nullptr, config.sharedMem, config.stream));
    }

    template <typename F, typename... Args>
#if __cplusplus >= 201703L
    std::enable_if_t<std::is_invocable_r<void, F, Args&&...>::value>
#else
    std::enable_if_t<std::is_void<std::result_of_t<F && (Args && ...)> >::value>
#endif
    launch_cooperative(F* kernel, LaunchParameters config, Args&&... args) {
      using function_type = detail::kernel_traits<F>;
      typename function_type::argument_type_tuple args_copy(args...);

      constexpr auto size = function_type::arguments_size;
      void const* pointers[size];

      detail::pointer_setter<size>()(pointers, args_copy);
      /*
      DPCT1004:75: Could not generate replacement.
      */
      cudaCheck(cudaLaunchCooperativeKernel(
          (const void*)kernel, config.gridDim, config.blockDim, (void**)pointers, config.sharedMem, config.stream));
    }

  }  // namespace sycltools
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_launch_h