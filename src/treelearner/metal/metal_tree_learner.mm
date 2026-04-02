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
  // For now, use CPU histogram construction (proven correct).
  // GPU acceleration will be enabled once kernel bin encoding is verified.
  // The GPU kernel dispatch infrastructure is ready — just need to align
  // the bin indexing between InitRowData (Feature4 packing) and
  // SerialTreeLearner's split evaluation.
  SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
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
