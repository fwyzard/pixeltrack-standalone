#ifndef HeterogeneousCore_CUDAUtilities_cuteCheck_h
#define HeterogeneousCore_CUDAUtilities_cuteCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <cuda.h>
#include <cuda.h>

namespace cms {
  namespace cuda {

    [[noreturn]] inline void abortOnCudaError(const char* file,
                                              int line,
                                              const char* cmd,
                                              const char* error,
                                              const char* message,
                                              const char* description = nullptr) {
      std::ostringstream out;
      out << "\n";
      out << file << ", line " << line << ":\n";
      out << "cuteCheck(" << cmd << ");\n";
      out << error << ": " << message << "\n";
      if (description)
        out << description << "\n";
      throw std::runtime_error(out.str());
    }

    inline bool cuteCheck_(
        const char* file, int line, const char* cmd, CUresult result, const char* description = nullptr) {
      if (result == CUDA_SUCCESS)
        return true;

      const char* error;
      cuGetErrorName(result, &error);
      const char* message;
      cuGetErrorString(result, &message);
      abortOnCudaError(file, line, cmd, error, message, description);
      return false;
    }

    inline bool cuteCheck_(
        const char* file, int line, const char* cmd, cudaError_t result, const char* description = nullptr) {
      if (result == cudaSuccess)
        return true;

      const char* error = cudaGetErrorName(result);
      const char* message = cudaGetErrorString(result);
      abortOnCudaError(file, line, cmd, error, message, description);
      return false;
    }

  }  // namespace cuda
}  // namespace cms

#define cuteCheck(ARG, ...) (cms::cuda::cuteCheck_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // HeterogeneousCore_CUDAUtilities_cuteCheck_h
