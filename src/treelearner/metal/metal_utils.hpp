/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_UTILS_HPP_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_UTILS_HPP_

#ifdef LGBM_USE_METAL

#include <LightGBM/utils/log.h>
#include <LightGBM/meta.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace LightGBM {

#define METAL_CHECK(condition, msg) \
  do { \
    if (!(condition)) { \
      LightGBM::Log::Fatal("[Metal] %s (%s:%d)", (msg), __FILE__, __LINE__); \
    } \
  } while (0)

/*!
 * \brief Singleton providing access to the Metal device, command queue,
 *        shader library, and cached compute pipeline states.
 */
class MetalDevice {
 public:
  /*! \brief Get the default Metal device (cached). */
  static void* GetDevice();

  /*! \brief Get a command queue for the default device (cached). */
  static void* GetQueue();

  /*! \brief Get the compiled metallib shader library (cached). */
  static void* GetLibrary();

  /*!
   * \brief Get (or create and cache) a compute pipeline state for the named
   *        kernel function.
   * \param function_name Name of the kernel in the metallib.
   * \return An id<MTLComputePipelineState> cast to void*.
   */
  static void* GetPipeline(const char* function_name);

  /*! \brief Find the path to lib_lightgbm.metallib. Returns NSString* as void*. */
  static void* FindMetallibPath();

 private:
  MetalDevice() = delete;
};

/*!
 * \brief RAII wrapper around an id<MTLBuffer> with StorageModeShared.
 *
 * Because the buffer lives in unified memory the CPU and GPU can both
 * access data() without explicit copies.
 *
 * Move-only; copies are deleted.
 */
template <typename T>
class MetalBuffer {
 public:
  MetalBuffer() : mtl_buffer_(nullptr), size_(0) {}

  explicit MetalBuffer(size_t n) : mtl_buffer_(nullptr), size_(0) {
    if (n > 0) {
      Allocate(n);
    }
  }

  ~MetalBuffer() {
    Release();
  }

  MetalBuffer(const MetalBuffer&) = delete;
  MetalBuffer& operator=(const MetalBuffer&) = delete;

  MetalBuffer(MetalBuffer&& other) noexcept
      : mtl_buffer_(other.mtl_buffer_), size_(other.size_) {
    other.mtl_buffer_ = nullptr;
    other.size_ = 0;
  }

  MetalBuffer& operator=(MetalBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      mtl_buffer_ = other.mtl_buffer_;
      size_ = other.size_;
      other.mtl_buffer_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /*! \brief Pointer to the shared-memory contents (CPU + GPU accessible). */
  T* data() const;

  /*! \brief Number of T elements in the buffer. */
  size_t size() const { return size_; }

  /*!
   * \brief Resize the buffer. Reallocates only when \p n differs from the
   *        current size. Existing contents are NOT preserved.
   */
  void Resize(size_t n) {
    if (n == size_) {
      return;
    }
    Release();
    if (n > 0) {
      Allocate(n);
    }
  }

  /*! \brief The underlying id<MTLBuffer> as a void*. */
  void* GetMTLBuffer() const { return mtl_buffer_; }

 private:
  void Allocate(size_t n);
  void Release();

  void* mtl_buffer_;
  size_t size_;
};

// ---- Template method definitions (declared out-of-line) ----
// The implementations are in metal_utils.mm via explicit instantiations
// for the types actually used.  For header-only consumption by .cpp files
// that do NOT link Objective-C, the linker resolves them from the .mm TU.

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_METAL_UTILS_HPP_
