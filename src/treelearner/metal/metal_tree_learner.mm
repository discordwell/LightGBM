/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef LGBM_USE_METAL

#include "metal_tree_learner.hpp"
#include "metal_utils.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/utils/array_args.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "../../io/dense_bin.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace LightGBM {

// ============================================================================
// Constructor / Destructor
// ============================================================================

MetalSingleGPUTreeLearner::MetalSingleGPUTreeLearner(const Config* config)
    : SerialTreeLearner(config) {
}

MetalSingleGPUTreeLearner::~MetalSingleGPUTreeLearner() {
  @autoreleasepool {
    if (pending_command_buffer_) {
      (void)(__bridge_transfer id<MTLCommandBuffer>)pending_command_buffer_;
    }
    if (histogram_output_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)histogram_output_buffer_;
    }
    if (data_indices_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)data_indices_buffer_;
    }
    if (hessians_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)hessians_buffer_;
    }
    if (gradients_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)gradients_buffer_;
    }
    if (features_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)features_buffer_;
    }
    if (histogram_pipeline_) {
      (void)(__bridge_transfer id<MTLComputePipelineState>)histogram_pipeline_;
    }
    if (metal_library_) {
      (void)(__bridge_transfer id<MTLLibrary>)metal_library_;
    }
    if (metal_queue_) {
      (void)(__bridge_transfer id<MTLCommandQueue>)metal_queue_;
    }
    // Device is a system singleton — don't release
  }
}

// ============================================================================
// Init
// ============================================================================

void MetalSingleGPUTreeLearner::Init(const Dataset* train_data,
                                     bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_feature_groups_ = train_data_->num_feature_groups();
  InitMetal();
}

void MetalSingleGPUTreeLearner::InitMetal() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    METAL_CHECK(device != nil, "No Metal device found");
    metal_device_ = (__bridge_retained void*)device;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    METAL_CHECK(queue != nil, "Failed to create Metal command queue");
    metal_queue_ = (__bridge_retained void*)queue;

    // Load metallib
    id<MTLLibrary> library = nil;
    NSError* error = nil;
    NSString* path = (__bridge NSString*)MetalDevice::FindMetallibPath();
    METAL_CHECK(path != nil, "Cannot find lib_lightgbm.metallib");
    NSURL* url = [NSURL fileURLWithPath:path];
    library = [device newLibraryWithURL:url error:&error];
    METAL_CHECK(library != nil,
                error ? [[error localizedDescription] UTF8String] : "unknown");
    metal_library_ = (__bridge_retained void*)library;

    // Determine kernel variant based on max_num_bin
    max_num_bin_ = 0;
    for (int i = 0; i < num_feature_groups_; ++i) {
      max_num_bin_ = std::max(max_num_bin_,
                              train_data_->FeatureGroupNumBin(i));
    }

    if (max_num_bin_ <= 16) {
      kernel_name_ = "histogram16";
      device_bin_size_ = 16;
      dword_features_ = 8;
    } else if (max_num_bin_ <= 64) {
      kernel_name_ = "histogram64";
      device_bin_size_ = 64;
      dword_features_ = 4;
    } else if (max_num_bin_ <= 256) {
      kernel_name_ = "histogram256";
      device_bin_size_ = 256;
      dword_features_ = 4;
    } else {
      Log::Fatal("bin size %d cannot run on Metal GPU (max 256)", max_num_bin_);
    }

    // Pipeline creation deferred until GPU histogram is enabled.
    // Currently using CPU histogram construction via SerialTreeLearner.
    hist_bin_entry_sz_ = sizeof(gpu_hist_t) * 2;  // FP32 grad + hess
  }

  AllocateMetalBuffers();
}

// ============================================================================
// Train — delegates to SerialTreeLearner, which calls our ConstructHistograms
// ============================================================================

Tree* MetalSingleGPUTreeLearner::Train(const score_t* gradients,
                                       const score_t* hessians,
                                       bool is_first_tree) {
  return SerialTreeLearner::Train(gradients, hessians, is_first_tree);
}

// ============================================================================
// BeforeTrain — copy gradients/hessians to Metal buffers early
// ============================================================================

void MetalSingleGPUTreeLearner::BeforeTrain() {
  @autoreleasepool {
    // Copy gradients to Metal shared buffer (unified memory — fast memcpy)
    id<MTLBuffer> gradBuf = (__bridge id<MTLBuffer>)gradients_buffer_;
    std::memcpy([gradBuf contents], gradients_, num_data_ * sizeof(score_t));

    if (!share_state_->is_constant_hessian) {
      id<MTLBuffer> hessBuf = (__bridge id<MTLBuffer>)hessians_buffer_;
      std::memcpy([hessBuf contents], hessians_, num_data_ * sizeof(score_t));
    }
  }
  SerialTreeLearner::BeforeTrain();
}

// ============================================================================
// ConstructHistograms — GPU accelerated (overrides SerialTreeLearner)
// ============================================================================

void MetalSingleGPUTreeLearner::ConstructHistograms(
    const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  // Build histogram for the smaller leaf using Metal GPU.
  hist_t* ptr_smaller_leaf_hist_data =
      smaller_leaf_histogram_array_[0].RawData() - kHistOffset;

  const data_size_t num_data_in_leaf = smaller_leaf_splits_->num_data_in_leaf();
  const data_size_t* data_indices = smaller_leaf_splits_->data_indices();
  const int num_groups = train_data_->num_feature_groups();
  const int total_bins = train_data_->NumTotalBin();

  // Pack bin data: row-major [num_data × num_groups] uint8
  // Uses unified memory — CPU writes, GPU reads, zero copy cost.
  if (!bin_data_packed_) {
    const size_t pack_size = static_cast<size_t>(num_data_) * num_groups;
    @autoreleasepool {
      id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device_;
      id<MTLBuffer> buf = [dev newBufferWithLength:pack_size
                                           options:MTLResourceStorageModeShared];
      bin_data_buffer_ = (__bridge_retained void*)buf;
    }
    uint8_t* dst = reinterpret_cast<uint8_t*>(
        [(__bridge id<MTLBuffer>)bin_data_buffer_ contents]);

    // Pack using feature group iterators (same bin values as CPU histogram)
    for (int g = 0; g < num_groups; ++g) {
      if (train_data_->IsMultiGroup(g)) {
        // Multi-valued groups: skip for GPU, will be handled by CPU below
        for (data_size_t row = 0; row < num_data_; ++row) {
          dst[static_cast<size_t>(row) * num_groups + g] = 0;
        }
        continue;
      }
      BinIterator* iter = train_data_->FeatureGroupIterator(g);
      iter->Reset(0);
      for (data_size_t row = 0; row < num_data_; ++row) {
        dst[static_cast<size_t>(row) * num_groups + g] =
            static_cast<uint8_t>(iter->RawGet(row));
      }
    }
    bin_data_packed_ = true;

    // Build group bin offset array
    group_bin_offsets_.resize(num_groups + 1);
    group_bin_offsets_[0] = 0;
    for (int g = 0; g < num_groups; ++g) {
      group_bin_offsets_[g + 1] =
          static_cast<uint32_t>(train_data_->GroupBinBoundary(g + 1));
    }
    @autoreleasepool {
      id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device_;
      id<MTLBuffer> buf = [dev newBufferWithBytes:group_bin_offsets_.data()
                                          length:group_bin_offsets_.size() * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
      group_offsets_buffer_ = (__bridge_retained void*)buf;
    }

    // Create histogram_simple pipeline
    @autoreleasepool {
      id<MTLLibrary> lib = (__bridge id<MTLLibrary>)metal_library_;
      id<MTLFunction> func = [lib newFunctionWithName:@"histogram_grouped"];
      METAL_CHECK(func != nil, "histogram_simple kernel not found");
      NSError* error = nil;
      id<MTLComputePipelineState> pso =
          [(__bridge id<MTLDevice>)metal_device_
              newComputePipelineStateWithFunction:func error:&error];
      METAL_CHECK(pso != nil, "Failed to create histogram_simple pipeline");
      histogram_pipeline_ = (__bridge_retained void*)pso;
    }

    // Float histogram buffer
    @autoreleasepool {
      id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device_;
      id<MTLBuffer> buf = [dev newBufferWithLength:total_bins * 2 * sizeof(float)
                                           options:MTLResourceStorageModeShared];
      histogram_output_buffer_ = (__bridge_retained void*)buf;
    }
  }

  // --- GPU histogram dispatch ---
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_queue_;
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)histogram_pipeline_;
    [enc setComputePipelineState:pso];

    // Zero the float histogram
    float* hist_float = reinterpret_cast<float*>(
        [(__bridge id<MTLBuffer>)histogram_output_buffer_ contents]);
    std::memset(hist_float, 0, total_bins * 2 * sizeof(float));

    // Set buffers
    [enc setBuffer:(__bridge id<MTLBuffer>)gradients_buffer_ offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)hessians_buffer_ offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bin_data_buffer_ offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)group_offsets_buffer_ offset:0 atIndex:3];

    // Data indices — need to pass them in a Metal buffer
    id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device_;
    id<MTLBuffer> idx_buf;
    if (num_data_in_leaf == num_data_) {
      // Root node: sequential indices
      idx_buf = (__bridge id<MTLBuffer>)data_indices_buffer_;
      int* idx_ptr = reinterpret_cast<int*>([idx_buf contents]);
      for (data_size_t i = 0; i < num_data_; ++i) idx_ptr[i] = i;
    } else {
      idx_buf = (__bridge id<MTLBuffer>)data_indices_buffer_;
      std::memcpy([idx_buf contents], data_indices,
                  num_data_in_leaf * sizeof(data_size_t));
    }
    [enc setBuffer:idx_buf offset:0 atIndex:4];

    [enc setBuffer:(__bridge id<MTLBuffer>)histogram_output_buffer_ offset:0 atIndex:5];

    uint32_t num_data_arg = static_cast<uint32_t>(num_data_in_leaf);
    uint32_t num_groups_arg = static_cast<uint32_t>(num_groups);
    uint32_t total_bins_arg = static_cast<uint32_t>(total_bins);
    [enc setBytes:&num_data_arg length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&num_groups_arg length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&total_bins_arg length:sizeof(uint32_t) atIndex:8];

    // Dispatch
    // One threadgroup per feature group, threads iterate over data
    NSUInteger tg_size = std::min(256u, (uint32_t)[pso maxTotalThreadsPerThreadgroup]);
    NSUInteger num_tg = static_cast<NSUInteger>(num_groups);
    [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Convert float histogram → double (hist_t) into the output array
    for (int g = 0; g < num_groups; ++g) {
      if (train_data_->IsMultiGroup(g)) continue;
      const int start = train_data_->GroupBinBoundary(g);
      const int end = train_data_->GroupBinBoundary(g + 1);
      hist_t* dst = ptr_smaller_leaf_hist_data + start * 2;
      const float* src = hist_float + start * 2;
      for (int b = start; b < end; ++b) {
        dst[(b - start) * 2] = static_cast<hist_t>(src[(b) * 2 - start * 2]);
        dst[(b - start) * 2 + 1] = static_cast<hist_t>(src[(b) * 2 + 1 - start * 2]);
      }
    }
  }

  // Handle sparse feature groups on CPU
  {
    std::vector<int8_t> is_sparse_used(num_features_, 0);
    for (int f = 0; f < num_features_; ++f) {
      if (!is_feature_used[f]) continue;
      if (train_data_->IsMultiGroup(train_data_->Feature2Group(f))) {
        is_sparse_used[f] = 1;
      }
    }
    train_data_->ConstructHistograms<false, 0>(
        is_sparse_used, data_indices, num_data_in_leaf,
        gradients_, hessians_,
        ordered_gradients_.data(), ordered_hessians_.data(),
        share_state_.get(), ptr_smaller_leaf_hist_data);
  }

  // Handle larger leaf (subtraction or explicit)
  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    hist_t* ptr_larger = larger_leaf_histogram_array_[0].RawData() - kHistOffset;
    // Use CPU for the larger leaf for now (TODO: GPU acceleration)
    train_data_->ConstructHistograms<false, 0>(
        is_feature_used, larger_leaf_splits_->data_indices(),
        larger_leaf_splits_->num_data_in_leaf(),
        gradients_, hessians_,
        ordered_gradients_.data(), ordered_hessians_.data(),
        share_state_.get(), ptr_larger);
  }
}

// ============================================================================
// AllocateMetalBuffers — pack Feature4 data, allocate GPU buffers
// ============================================================================

void MetalSingleGPUTreeLearner::AllocateMetalBuffers() {
  // Count dense feature groups
  num_dense_feature_groups_ = 0;
  for (int i = 0; i < num_feature_groups_; ++i) {
    if (!train_data_->IsMultiGroup(i)) {
      num_dense_feature_groups_++;
    }
  }
  num_dense_feature4_ = (num_dense_feature_groups_ + (dword_features_ - 1)) / dword_features_;

  if (!num_dense_feature_groups_) {
    Log::Warning("Metal GPU acceleration disabled — no dense features found");
    return;
  }

  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)metal_device_;
    int allocated_num_data = num_data_ + 256 * (1 << 10);  // prefetch margin

    // Gradient and hessian buffers
    if (gradients_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)gradients_buffer_;
    }
    id<MTLBuffer> gradBuf = [device newBufferWithLength:allocated_num_data * sizeof(score_t)
                                               options:MTLResourceStorageModeShared];
    gradients_buffer_ = (__bridge_retained void*)gradBuf;

    if (hessians_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)hessians_buffer_;
    }
    id<MTLBuffer> hessBuf = [device newBufferWithLength:allocated_num_data * sizeof(score_t)
                                               options:MTLResourceStorageModeShared];
    hessians_buffer_ = (__bridge_retained void*)hessBuf;

    // Data indices buffer
    if (data_indices_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)data_indices_buffer_;
    }
    id<MTLBuffer> idxBuf = [device newBufferWithLength:allocated_num_data * sizeof(data_size_t)
                                              options:MTLResourceStorageModeShared];
    data_indices_buffer_ = (__bridge_retained void*)idxBuf;

    // Histogram output buffer (float, converted to double when read)
    size_t hist_output_size = num_dense_feature4_ * dword_features_ * device_bin_size_ * hist_bin_entry_sz_;
    if (histogram_output_buffer_) {
      (void)(__bridge_transfer id<MTLBuffer>)histogram_output_buffer_;
    }
    id<MTLBuffer> histBuf = [device newBufferWithLength:hist_output_size
                                               options:MTLResourceStorageModeShared];
    histogram_output_buffer_ = (__bridge_retained void*)histBuf;
  }
}

// ============================================================================
// ResetTrainingData
// ============================================================================

void MetalSingleGPUTreeLearner::ResetTrainingData(
    const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  num_feature_groups_ = train_data_->num_feature_groups();
  AllocateMetalBuffers();
}

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
