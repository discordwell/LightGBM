/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef LGBM_USE_METAL

#include "metal_best_split_finder.hpp"
#include "metal_leaf_splits.hpp"
#include "metal_tree_learner.hpp"
#include "metal_utils.hpp"

#include "../cost_effective_gradient_boosting.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/network.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace LightGBM {

static constexpr int kGatherThreshold = 64;
static constexpr uint32_t kMaxSubhistParts = 16;
static constexpr uint32_t kTargetRowsPerSubhist = 8192;
static constexpr size_t kSmallerLeafSlot = 0;
static constexpr size_t kLargerLeafSlot = 1;

struct PackedFeatureTuple {
  uint8_t bins[4];
};
static_assert(sizeof(PackedFeatureTuple) == 4,
              "PackedFeatureTuple must match Metal uchar4 layout");

MetalSingleGPUTreeLearner::MetalSingleGPUTreeLearner(const Config* config)
    : SerialTreeLearner(config) {
}

MetalSingleGPUTreeLearner::~MetalSingleGPUTreeLearner() {
  @autoreleasepool {
    auto release = [](void*& p) {
      if (p) { CFRelease(p); p = nullptr; }
    };
    release(subhist_buffer_);
    release(ordered_packed_bins_buffer_);
    release(ordered_bins_buffer_);
    release(ordered_hess_buffer_);
    release(ordered_grad_buffer_);
    release(partition_output_buffer_);
    release(partition_counts_buffer_);
    release(histogram_output_buffer_);
    release(data_indices_buffer_);
    release(hessians_buffer_);
    release(gradients_buffer_);
    release(bin_data_packed_buffer_);
    release(bin_data_col_buffer_);
    release(bin_data_row_buffer_);
    release(dense_group_map_buffer_);
    release(group_offsets_buffer_);
    release(partition_pipeline_);
    release(packed_gather_pipeline_);
    release(packed_reduction_pipeline_);
    release(packed_histogram_pipeline_);
    release(gather_pipeline_);
    release(histogram_row_pipeline_);
    release(reduction_pipeline_);
    release(histogram_pipeline_);
    release(metal_library_);
    release(metal_queue_);
    // metal_device_ is a system singleton — don't release
  }
}

void MetalSingleGPUTreeLearner::Init(const Dataset* train_data,
                                     bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  metal_leaf_splits_.reset(
      new MetalLeafSplits(num_data_, static_cast<size_t>(config_->num_leaves)));
  ValidateTrainingScope(train_data_);
  num_feature_groups_ = train_data_->num_feature_groups();
  best_split_finder_.reset(new MetalBestSplitFinder(
      train_data_, share_state_->feature_hist_offsets(), config_));
  best_split_finder_->Init();
  InitMetal();
}

void MetalSingleGPUTreeLearner::ValidateTrainingScope(
    const Dataset* train_data) const {
  if (config_->task != TaskType::kTrain) {
    Log::Fatal("Metal tree learner only supports task=train.");
  }
  if (config_->tree_learner != std::string("serial")) {
    Log::Fatal("Metal tree learner only supports tree_learner=serial.");
  }
  if (config_->boosting != std::string("gbdt")) {
    Log::Fatal("Metal tree learner only supports boosting=gbdt.");
  }
  if (Network::num_machines() != 1) {
    Log::Fatal("Metal tree learner does not support distributed training.");
  }
  if (config_->max_bin > 256) {
    Log::Fatal("Metal tree learner only supports max_bin <= 256.");
  }
  if (config_->use_quantized_grad) {
    Log::Fatal("Metal tree learner does not support quantized training.");
  }
  if (config_->linear_tree) {
    Log::Fatal("Metal tree learner does not support linear trees.");
  }
  if (config_->num_class != 1) {
    Log::Fatal("Metal tree learner only supports single-output objectives.");
  }
  if (config_->objective == std::string("multiclass") ||
      config_->objective == std::string("multiclassova") ||
      config_->objective == std::string("lambdarank") ||
      config_->objective == std::string("rank_xendcg") ||
      config_->objective == std::string("cross_entropy") ||
      config_->objective == std::string("cross_entropy_lambda")) {
    Log::Fatal("Metal tree learner only supports binary and single-output regression objectives.");
  }
  if (config_->extra_trees) {
    Log::Fatal("Metal tree learner does not support extra_trees.");
  }
  if (config_->path_smooth > 0.0) {
    Log::Fatal("Metal tree learner does not support path_smooth.");
  }
  if (config_->max_delta_step > 0.0) {
    Log::Fatal("Metal tree learner does not support max_delta_step.");
  }
  if (!config_->monotone_constraints.empty()) {
    Log::Fatal("Metal tree learner does not support monotone constraints.");
  }
  if (!config_->feature_contri.empty()) {
    Log::Fatal("Metal tree learner does not support feature penalties.");
  }
  if (!config_->forcedsplits_filename.empty()) {
    Log::Fatal("Metal tree learner does not support forced splits.");
  }
  if (config_->cegb_penalty_split != 0.0 ||
      config_->cegb_tradeoff != 1.0 ||
      !config_->cegb_penalty_feature_lazy.empty() ||
      !config_->cegb_penalty_feature_coupled.empty()) {
    Log::Fatal("Metal tree learner does not support CEGB penalties.");
  }
  for (int feature_index = 0; feature_index < train_data->num_features();
       ++feature_index) {
    if (train_data->FeatureBinMapper(feature_index)->bin_type() !=
        BinType::NumericalBin) {
      Log::Fatal("Metal tree learner only supports dense numerical features.");
    }
  }
  for (int group = 0; group < train_data->num_feature_groups(); ++group) {
    if (train_data->IsMultiGroup(group)) {
      Log::Fatal("Metal tree learner does not support sparse or multi-group feature groups.");
    }
  }
}

void MetalSingleGPUTreeLearner::InitMetal() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    METAL_CHECK(device != nil, "No Metal device found");
    metal_device_ = (__bridge_retained void*)device;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    METAL_CHECK(queue != nil, "Failed to create Metal command queue");
    metal_queue_ = (__bridge_retained void*)queue;

    NSError* error = nil;
    NSString* path = (__bridge NSString*)MetalDevice::FindMetallibPath();
    METAL_CHECK(path != nil, "Cannot find lib_lightgbm.metallib");
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                                error:&error];
    METAL_CHECK(library != nil,
                error ? [[error localizedDescription] UTF8String] : "unknown");
    metal_library_ = (__bridge_retained void*)library;

    max_num_bin_ = 0;
    for (int i = 0; i < num_feature_groups_; ++i) {
      max_num_bin_ = std::max(max_num_bin_,
                              train_data_->FeatureGroupNumBin(i));
    }
    if (max_num_bin_ > 256) {
      Log::Fatal("bin size %d cannot run on Metal GPU (max 256)", max_num_bin_);
    }

    num_dense_feature_groups_ = 0;
    for (int i = 0; i < num_feature_groups_; ++i) {
      if (!train_data_->IsMultiGroup(i)) num_dense_feature_groups_++;
    }
    num_dense_feature_tuples_ = (num_dense_feature_groups_ + 3) / 4;
    use_row_parallel_ = (num_dense_feature_groups_ >= kGatherThreshold);
  }
  AllocateMetalBuffers();
}

const MetalLeafSplitsStruct* MetalSingleGPUTreeLearner::GetActiveMetalLeafState(
    size_t slot) const {
  const LeafSplits* leaf_splits =
      slot == kSmallerLeafSlot ? smaller_leaf_splits_.get() : larger_leaf_splits_.get();
  if (leaf_splits == nullptr || leaf_splits->leaf_index() < 0) {
    return nullptr;
  }
  return FindActiveMetalLeafState(leaf_splits->leaf_index());
}

const MetalLeafSplitsStruct* MetalSingleGPUTreeLearner::FindActiveMetalLeafState(
    int leaf_index) const {
  if (metal_leaf_splits_ == nullptr || leaf_index < 0 ||
      static_cast<size_t>(leaf_index) >= metal_leaf_splits_->num_slots()) {
    return nullptr;
  }
  const MetalLeafSplitsStruct* leaf_state =
      metal_leaf_splits_->GetStruct(static_cast<size_t>(leaf_index));
  return leaf_state->leaf_index == leaf_index ? leaf_state : nullptr;
}

void MetalSingleGPUTreeLearner::ResetMetalLeafStateTable() {
  if (metal_leaf_splits_ == nullptr) {
    metal_leaf_splits_.reset(
        new MetalLeafSplits(num_data_, static_cast<size_t>(config_->num_leaves)));
  } else {
    metal_leaf_splits_->ResizeSlots(static_cast<size_t>(config_->num_leaves));
  }
  metal_leaf_splits_->Reset();
}

void MetalSingleGPUTreeLearner::SyncMetalLeafState(int leaf_index,
                                                   const LeafSplits* leaf_splits) {
  if (leaf_index < 0 || leaf_splits == nullptr || metal_leaf_splits_ == nullptr) {
    return;
  }
  CHECK_EQ(leaf_splits->leaf_index(), leaf_index);
  metal_leaf_splits_->SyncLeaf(static_cast<size_t>(leaf_index), leaf_splits,
                               data_partition_.get());
}

void MetalSingleGPUTreeLearner::UpdateMetalLeafState(
    int leaf_index,
    double sum_gradients,
    double sum_hessians,
    data_size_t num_data_in_leaf,
    double leaf_value) {
  if (leaf_index < 0 || metal_leaf_splits_ == nullptr) {
    return;
  }
  metal_leaf_splits_->UpdateLeaf(static_cast<size_t>(leaf_index), leaf_index,
                                 sum_gradients, sum_hessians, num_data_in_leaf,
                                 leaf_value, data_partition_->leaf_begin(leaf_index));
}

void MetalSingleGPUTreeLearner::SyncMetalActiveLeafState() {
  if (smaller_leaf_splits_ != nullptr && smaller_leaf_splits_->leaf_index() >= 0) {
    SyncMetalLeafState(smaller_leaf_splits_->leaf_index(), smaller_leaf_splits_.get());
  }
  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->leaf_index() >= 0) {
    SyncMetalLeafState(larger_leaf_splits_->leaf_index(), larger_leaf_splits_.get());
  }
}

double MetalSingleGPUTreeLearner::GetMetalParentOutput(
    const Tree* tree,
    const MetalLeafSplitsStruct* leaf_state) const {
  if (leaf_state == nullptr || leaf_state->leaf_index < 0) {
    return 0.0;
  }
  if (tree->num_leaves() == 1) {
    return FeatureHistogram::CalculateSplittedLeafOutput<true, true, true, false>(
        leaf_state->sum_of_gradients, leaf_state->sum_of_hessians,
        config_->lambda_l1, config_->lambda_l2, config_->max_delta_step,
        BasicConstraint(), config_->path_smooth, leaf_state->num_data_in_leaf,
        0);
  }
  return leaf_state->leaf_value;
}

Tree* MetalSingleGPUTreeLearner::Train(const score_t* gradients,
                                       const score_t* hessians,
                                       bool is_first_tree) {
  return SerialTreeLearner::Train(gradients, hessians, is_first_tree);
}

Tree* MetalSingleGPUTreeLearner::FitByExistingTree(
    const Tree* /*old_tree*/,
    const score_t* /*gradients*/,
    const score_t* /*hessians*/) const {
  Log::Fatal("Metal tree learner does not support refit paths.");
  return nullptr;
}

Tree* MetalSingleGPUTreeLearner::FitByExistingTree(
    const Tree* /*old_tree*/,
    const std::vector<int>& /*leaf_pred*/,
    const score_t* /*gradients*/,
    const score_t* /*hessians*/) const {
  Log::Fatal("Metal tree learner does not support refit paths.");
  return nullptr;
}

void MetalSingleGPUTreeLearner::BeforeTrain() {
  if (forced_split_json_ != nullptr) {
    Log::Fatal("Metal tree learner does not support forced splits.");
  }
  @autoreleasepool {
    id<MTLBuffer> gradBuf = (__bridge id<MTLBuffer>)gradients_buffer_;
    std::memcpy([gradBuf contents], gradients_, num_data_ * sizeof(score_t));
    if (hessians_ != nullptr) {
      id<MTLBuffer> hessBuf = (__bridge id<MTLBuffer>)hessians_buffer_;
      std::memcpy([hessBuf contents], hessians_, num_data_ * sizeof(score_t));
    }
  }
  SerialTreeLearner::BeforeTrain();
  ResetMetalLeafStateTable();
  SyncMetalActiveLeafState();
  if (best_split_finder_ != nullptr) {
    best_split_finder_->BeforeTrain(col_sampler_.is_feature_used_bytree());
  }
}

// ============================================================================
// ConstructHistograms
// ============================================================================

void MetalSingleGPUTreeLearner::ConstructHistograms(
    const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  Common::FunctionTimer fun_timer(
      "MetalSingleGPUTreeLearner::ConstructHistograms", global_timer);
  const bool stage_timing =
      std::getenv("LIGHTGBM_METAL_STAGE_TIMING") != nullptr;
  auto stage_start = std::chrono::steady_clock::now();
  auto log_stage = [&](const char* name) {
    if (!stage_timing) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(
        now - stage_start).count();
    Log::Info("[MetalTiming] %s took %.3f ms", name, ms);
    stage_start = now;
  };
  hist_t* ptr_smaller_leaf_hist_data =
      smaller_leaf_histogram_array_[0].RawData() - kHistOffset;

  const MetalLeafSplitsStruct* smaller_leaf =
      GetActiveMetalLeafState(kSmallerLeafSlot);
  const MetalLeafSplitsStruct* larger_leaf =
      GetActiveMetalLeafState(kLargerLeafSlot);
  CHECK(smaller_leaf != nullptr);
  const data_size_t num_data_in_leaf = smaller_leaf->num_data_in_leaf;
  const data_size_t* data_indices =
      data_partition_->indices() + smaller_leaf->data_indices_offset;
  const int num_groups = train_data_->num_feature_groups();
  const int total_bins = train_data_->NumTotalBin();

  // One-time setup: pack bin data and create pipelines
  if (!bin_data_packed_) {
    @autoreleasepool {
      id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device_;
      const size_t pack_size = static_cast<size_t>(num_data_) * num_groups;
      auto replace = [](void*& ptr, id obj) {
        if (ptr) {
          CFRelease(ptr);
        }
        ptr = (__bridge_retained void*)obj;
      };

      // Column-major bin data (used by both paths)
      id<MTLBuffer> colBuf = [dev newBufferWithLength:pack_size
                                              options:MTLResourceStorageModeShared];
      replace(bin_data_col_buffer_, colBuf);
      uint8_t* dst = reinterpret_cast<uint8_t*>([colBuf contents]);

      for (int g = 0; g < num_groups; ++g) {
        uint8_t* col = dst + static_cast<size_t>(g) * num_data_;
        if (train_data_->IsMultiGroup(g)) {
          std::memset(col, 0, num_data_);
          continue;
        }
        BinIterator* iter = train_data_->FeatureGroupIterator(g);
        iter->Reset(0);
        for (data_size_t row = 0; row < num_data_; ++row) {
          col[row] = static_cast<uint8_t>(iter->RawGet(row));
        }
      }

      if (num_dense_feature_tuples_ > 0) {
        const size_t packed_size = static_cast<size_t>(num_dense_feature_tuples_) *
                                   static_cast<size_t>(num_data_) *
                                   sizeof(PackedFeatureTuple);
        id<MTLBuffer> packedBuf = [dev newBufferWithLength:packed_size
                                                   options:MTLResourceStorageModeShared];
        replace(bin_data_packed_buffer_, packedBuf);
        PackedFeatureTuple* packed_dst =
            reinterpret_cast<PackedFeatureTuple*>([packedBuf contents]);
        std::memset(packed_dst, 0, packed_size);

        dense_group_map_.assign(static_cast<size_t>(num_dense_feature_tuples_) * 4,
                                0xFFFFFFFFu);
        for (int tuple = 0; tuple < num_dense_feature_tuples_; ++tuple) {
          for (int lane = 0; lane < 4; ++lane) {
            const int group = tuple * 4 + lane;
            if (group >= num_groups) {
              continue;
            }
            dense_group_map_[static_cast<size_t>(tuple) * 4 + lane] =
                static_cast<uint32_t>(group);
            const uint8_t* src_col =
                dst + static_cast<size_t>(group) * static_cast<size_t>(num_data_);
            for (data_size_t row = 0; row < num_data_; ++row) {
              packed_dst[static_cast<size_t>(tuple) * static_cast<size_t>(num_data_) +
                         static_cast<size_t>(row)]
                  .bins[lane] = src_col[row];
            }
          }
        }
        id<MTLBuffer> denseMapBuf =
            [dev newBufferWithBytes:dense_group_map_.data()
                             length:dense_group_map_.size() * sizeof(uint32_t)
                            options:MTLResourceStorageModeShared];
        replace(dense_group_map_buffer_, denseMapBuf);
      }

      // Group bin offsets
      group_bin_offsets_.resize(num_groups + 1);
      group_bin_offsets_[0] = 0;
      for (int g = 0; g < num_groups; ++g) {
        group_bin_offsets_[g + 1] =
            static_cast<uint32_t>(train_data_->GroupBinBoundary(g + 1));
      }
      id<MTLBuffer> offBuf = [dev newBufferWithBytes:group_bin_offsets_.data()
                                              length:group_bin_offsets_.size() * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
      replace(group_offsets_buffer_, offBuf);

      // Create pipelines
      id<MTLLibrary> lib = (__bridge id<MTLLibrary>)metal_library_;
      NSError* error = nil;

      // Column-grouped (fallback / narrow datasets)
      {
        id<MTLFunction> func = [lib newFunctionWithName:@"histogram_grouped"];
        METAL_CHECK(func != nil, "histogram_grouped kernel not found");
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:func error:&error];
        METAL_CHECK(pso != nil, "Failed to create grouped pipeline");
        replace(histogram_pipeline_, pso);
      }

      {
        id<MTLFunction> func = [lib newFunctionWithName:@"partition_indices_numeric"];
        METAL_CHECK(func != nil, "partition_indices_numeric kernel not found");
        error = nil;
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:func error:&error];
        METAL_CHECK(pso != nil, "Failed to create partition pipeline");
        replace(partition_pipeline_, pso);
      }

      if (use_row_parallel_) {
        id<MTLFunction> func = [lib newFunctionWithName:@"histogram_gathered_subhist"];
        METAL_CHECK(func != nil, "histogram_gathered_subhist kernel not found");
        error = nil;
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:func error:&error];
        METAL_CHECK(pso != nil, "Failed to create gathered subhist pipeline");
        replace(histogram_row_pipeline_, pso);

        id<MTLFunction> reduceFunc = [lib newFunctionWithName:@"reduce_histogram_subhist"];
        METAL_CHECK(reduceFunc != nil, "reduce_histogram_subhist kernel not found");
        error = nil;
        id<MTLComputePipelineState> reducePso =
            [dev newComputePipelineStateWithFunction:reduceFunc error:&error];
        METAL_CHECK(reducePso != nil, "Failed to create subhist reduction pipeline");
        replace(reduction_pipeline_, reducePso);

        id<MTLFunction> packedFunc = [lib newFunctionWithName:@"histogram_packed_subhist"];
        METAL_CHECK(packedFunc != nil, "histogram_packed_subhist kernel not found");
        error = nil;
        id<MTLComputePipelineState> packedPso =
            [dev newComputePipelineStateWithFunction:packedFunc error:&error];
        METAL_CHECK(packedPso != nil, "Failed to create packed subhist pipeline");
        replace(packed_histogram_pipeline_, packedPso);

        id<MTLFunction> packedReduceFunc =
            [lib newFunctionWithName:@"reduce_histogram_packed_subhist"];
        METAL_CHECK(packedReduceFunc != nil,
                    "reduce_histogram_packed_subhist kernel not found");
        error = nil;
        id<MTLComputePipelineState> packedReducePso =
            [dev newComputePipelineStateWithFunction:packedReduceFunc error:&error];
        METAL_CHECK(packedReducePso != nil,
                    "Failed to create packed subhist reduction pipeline");
        replace(packed_reduction_pipeline_, packedReducePso);

        // Gather kernel
        id<MTLFunction> gFunc = [lib newFunctionWithName:@"gather_to_leaf_order"];
        METAL_CHECK(gFunc != nil, "gather_to_leaf_order kernel not found");
        error = nil;
        id<MTLComputePipelineState> gPso =
            [dev newComputePipelineStateWithFunction:gFunc error:&error];
        METAL_CHECK(gPso != nil, "Failed to create gather pipeline");
        replace(gather_pipeline_, gPso);

        id<MTLFunction> packedGFunc =
            [lib newFunctionWithName:@"gather_packed_to_leaf_order"];
        METAL_CHECK(packedGFunc != nil, "gather_packed_to_leaf_order kernel not found");
        error = nil;
        id<MTLComputePipelineState> packedGPso =
            [dev newComputePipelineStateWithFunction:packedGFunc error:&error];
        METAL_CHECK(packedGPso != nil, "Failed to create packed gather pipeline");
        replace(packed_gather_pipeline_, packedGPso);

        // Ordered buffers
        replace(ordered_grad_buffer_, [dev newBufferWithLength:num_data_ * sizeof(float)
                                                        options:MTLResourceStorageModeShared]);
        replace(ordered_hess_buffer_, [dev newBufferWithLength:num_data_ * sizeof(float)
                                                        options:MTLResourceStorageModeShared]);
        replace(ordered_bins_buffer_, [dev newBufferWithLength:pack_size
                                                        options:MTLResourceStorageModeShared]);
        replace(ordered_packed_bins_buffer_,
                [dev newBufferWithLength:static_cast<size_t>(num_dense_feature_tuples_) *
                                        static_cast<size_t>(num_data_) *
                                        sizeof(PackedFeatureTuple)
                                options:MTLResourceStorageModeShared]);
        replace(subhist_buffer_, [dev newBufferWithLength:static_cast<size_t>(num_dense_feature_tuples_) *
                                                        kMaxSubhistParts * 4 * 256 * 2 *
                                                        sizeof(float)
                                                  options:MTLResourceStorageModeShared]);
      }

      // Histogram output
      replace(histogram_output_buffer_,
              [dev newBufferWithLength:total_bins * 2 * sizeof(float)
                               options:MTLResourceStorageModeShared]);
    }
    bin_data_packed_ = true;
  }
  log_stage("histogram/setup");

  // --- Dispatch ---
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_queue_;
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

    // Zero histogram
    id<MTLBuffer> histBuf = (__bridge id<MTLBuffer>)histogram_output_buffer_;
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    [blit fillBuffer:histBuf
               range:NSMakeRange(0, total_bins * 2 * sizeof(float))
               value:0];
    [blit endEncoding];

    // Copy data indices
    id<MTLBuffer> idx_buf = (__bridge id<MTLBuffer>)data_indices_buffer_;
    if (num_data_in_leaf == num_data_) {
      int* idx_ptr = reinterpret_cast<int*>([idx_buf contents]);
      for (data_size_t i = 0; i < num_data_; ++i) idx_ptr[i] = i;
    } else {
      std::memcpy([idx_buf contents], data_indices,
                  num_data_in_leaf * sizeof(data_size_t));
    }

    if (use_row_parallel_) {
      const uint32_t subhist_parts = std::min<uint32_t>(
          kMaxSubhistParts,
          std::max<uint32_t>(
              1, static_cast<uint32_t>((num_data_in_leaf + kTargetRowsPerSubhist - 1) /
                                       kTargetRowsPerSubhist)));

      // Pass 1: gather grad / hess and packed dense feature tuples into leaf order
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      id<MTLComputePipelineState> gPso =
          (__bridge id<MTLComputePipelineState>)packed_gather_pipeline_;
      [enc setComputePipelineState:gPso];
      [enc setBuffer:(__bridge id<MTLBuffer>)gradients_buffer_ offset:0 atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>)hessians_buffer_ offset:0 atIndex:1];
      [enc setBuffer:idx_buf offset:0 atIndex:2];
      [enc setBuffer:(__bridge id<MTLBuffer>)ordered_grad_buffer_ offset:0 atIndex:3];
      [enc setBuffer:(__bridge id<MTLBuffer>)ordered_hess_buffer_ offset:0 atIndex:4];
      [enc setBuffer:(__bridge id<MTLBuffer>)bin_data_packed_buffer_ offset:0 atIndex:5];
      [enc setBuffer:(__bridge id<MTLBuffer>)ordered_packed_bins_buffer_ offset:0 atIndex:6];
      uint32_t nd = static_cast<uint32_t>(num_data_in_leaf);
      uint32_t ndt = static_cast<uint32_t>(num_data_);
      uint32_t nt = static_cast<uint32_t>(num_dense_feature_tuples_);
      [enc setBytes:&nd length:sizeof(uint32_t) atIndex:7];
      [enc setBytes:&ndt length:sizeof(uint32_t) atIndex:8];
      [enc setBytes:&nt length:sizeof(uint32_t) atIndex:9];
      NSUInteger tg = std::min(256u, (uint32_t)[gPso maxTotalThreadsPerThreadgroup]);
      [enc dispatchThreadgroups:MTLSizeMake((num_data_in_leaf + tg - 1) / tg, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      log_stage("histogram/gather_packed");

      // Pass 2: build a sub-histogram per (packed tuple, row partition).
      id<MTLComputeCommandEncoder> enc2 = [cmdBuf computeCommandEncoder];
      id<MTLComputePipelineState> hPso =
          (__bridge id<MTLComputePipelineState>)packed_histogram_pipeline_;
      [enc2 setComputePipelineState:hPso];
      [enc2 setBuffer:(__bridge id<MTLBuffer>)ordered_grad_buffer_ offset:0 atIndex:0];
      [enc2 setBuffer:(__bridge id<MTLBuffer>)ordered_hess_buffer_ offset:0 atIndex:1];
      [enc2 setBuffer:(__bridge id<MTLBuffer>)ordered_packed_bins_buffer_ offset:0 atIndex:2];
      [enc2 setBuffer:(__bridge id<MTLBuffer>)dense_group_map_buffer_ offset:0 atIndex:3];
      [enc2 setBuffer:(__bridge id<MTLBuffer>)group_offsets_buffer_ offset:0 atIndex:4];
      [enc2 setBuffer:(__bridge id<MTLBuffer>)subhist_buffer_ offset:0 atIndex:5];
      [enc2 setBytes:&nd length:sizeof(uint32_t) atIndex:6];
      [enc2 setBytes:&nt length:sizeof(uint32_t) atIndex:7];
      [enc2 setBytes:&subhist_parts length:sizeof(uint32_t) atIndex:8];
      NSUInteger tg2 = std::min(256u, (uint32_t)[hPso maxTotalThreadsPerThreadgroup]);
      [enc2 dispatchThreadgroups:MTLSizeMake(nt * subhist_parts, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
      [enc2 endEncoding];
      log_stage("histogram/build_subhist");

      // Pass 3: reduce per-partition packed sub-histograms into the final output.
      id<MTLComputeCommandEncoder> enc3 = [cmdBuf computeCommandEncoder];
      id<MTLComputePipelineState> rPso =
          (__bridge id<MTLComputePipelineState>)packed_reduction_pipeline_;
      [enc3 setComputePipelineState:rPso];
      [enc3 setBuffer:(__bridge id<MTLBuffer>)subhist_buffer_ offset:0 atIndex:0];
      [enc3 setBuffer:(__bridge id<MTLBuffer>)dense_group_map_buffer_ offset:0 atIndex:1];
      [enc3 setBuffer:(__bridge id<MTLBuffer>)group_offsets_buffer_ offset:0 atIndex:2];
      [enc3 setBuffer:histBuf offset:0 atIndex:3];
      [enc3 setBytes:&nt length:sizeof(uint32_t) atIndex:4];
      [enc3 setBytes:&subhist_parts length:sizeof(uint32_t) atIndex:5];
      NSUInteger tg3 = std::min(256u, (uint32_t)[rPso maxTotalThreadsPerThreadgroup]);
      [enc3 dispatchThreadgroups:MTLSizeMake(nt, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tg3, 1, 1)];
      [enc3 endEncoding];
      log_stage("histogram/reduce_subhist");
    } else {
      // Column-grouped: one threadgroup per feature group (original path)
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      id<MTLComputePipelineState> pso =
          (__bridge id<MTLComputePipelineState>)histogram_pipeline_;
      [enc setComputePipelineState:pso];
      [enc setBuffer:(__bridge id<MTLBuffer>)gradients_buffer_ offset:0 atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>)hessians_buffer_ offset:0 atIndex:1];
      [enc setBuffer:(__bridge id<MTLBuffer>)bin_data_col_buffer_ offset:0 atIndex:2];
      [enc setBuffer:(__bridge id<MTLBuffer>)group_offsets_buffer_ offset:0 atIndex:3];
      [enc setBuffer:idx_buf offset:0 atIndex:4];
      [enc setBuffer:histBuf offset:0 atIndex:5];
      uint32_t nd = static_cast<uint32_t>(num_data_in_leaf);
      uint32_t ng = static_cast<uint32_t>(num_groups);
      uint32_t tb = static_cast<uint32_t>(total_bins);
      uint32_t ndt = static_cast<uint32_t>(num_data_);
      [enc setBytes:&nd length:sizeof(uint32_t) atIndex:6];
      [enc setBytes:&ng length:sizeof(uint32_t) atIndex:7];
      [enc setBytes:&tb length:sizeof(uint32_t) atIndex:8];
      [enc setBytes:&ndt length:sizeof(uint32_t) atIndex:9];
      NSUInteger tg = std::min(256u, (uint32_t)[pso maxTotalThreadsPerThreadgroup]);
      [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
      log_stage("histogram/grouped_encode");
    }

    [cmdBuf commit];

    [cmdBuf waitUntilCompleted];
    log_stage("histogram/gpu_wait");

    // Convert float → double
    const float* hist_float = reinterpret_cast<const float*>([histBuf contents]);
    for (int g = 0; g < num_groups; ++g) {
      if (train_data_->IsMultiGroup(g)) continue;
      const int start = train_data_->GroupBinBoundary(g);
      const int nbins = train_data_->GroupBinBoundary(g + 1) - start;
      hist_t* dst = ptr_smaller_leaf_hist_data + start * 2;
      const float* src = hist_float + start * 2;
      for (int b = 0; b < nbins; ++b) {
        dst[b * 2]     = static_cast<hist_t>(src[b * 2]);
        dst[b * 2 + 1] = static_cast<hist_t>(src[b * 2 + 1]);
      }
    }
    log_stage("histogram/copyout");
  }

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract &&
      larger_leaf != nullptr &&
      larger_leaf->leaf_index >= 0 && larger_leaf->num_data_in_leaf > 0) {
    hist_t* ptr_larger = larger_leaf_histogram_array_[0].RawData() - kHistOffset;
    train_data_->ConstructHistograms<false, 0>(
        is_feature_used,
        data_partition_->indices() + larger_leaf->data_indices_offset,
        larger_leaf->num_data_in_leaf,
        gradients_, hessians_,
        ordered_gradients_.data(), ordered_hessians_.data(),
        share_state_.get(), ptr_larger);
  }
}

void MetalSingleGPUTreeLearner::FindBestSplitsFromHistograms(
    const std::vector<int8_t>& is_feature_used,
    bool use_subtract,
    const Tree* tree) {
  Common::FunctionTimer fun_timer(
      "MetalSingleGPUTreeLearner::FindBestSplitsFromHistograms", global_timer);
  const bool enable_gpu_split =
      std::getenv("LIGHTGBM_METAL_ENABLE_GPU_SPLIT") != nullptr &&
      std::getenv("LIGHTGBM_METAL_DISABLE_GPU_SPLIT") == nullptr;
  if (!enable_gpu_split) {
    SerialTreeLearner::FindBestSplitsFromHistograms(is_feature_used, use_subtract,
                                                    tree);
    return;
  }
  const bool debug_logging = std::getenv("LIGHTGBM_METAL_DEBUG") != nullptr;
  const bool stage_timing =
      std::getenv("LIGHTGBM_METAL_STAGE_TIMING") != nullptr;
  auto stage_start = std::chrono::steady_clock::now();
  auto log_stage = [&](const char* name) {
    if (!stage_timing) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(
        now - stage_start).count();
    Log::Info("[MetalTiming] %s took %.3f ms", name, ms);
    stage_start = now;
  };

  const MetalLeafSplitsStruct* smaller_leaf =
      GetActiveMetalLeafState(kSmallerLeafSlot);
  const MetalLeafSplitsStruct* larger_leaf =
      GetActiveMetalLeafState(kLargerLeafSlot);
  CHECK(smaller_leaf != nullptr);
  const int smaller_leaf_index = smaller_leaf->leaf_index;
  const int larger_leaf_index = larger_leaf != nullptr ? larger_leaf->leaf_index : -1;
  std::vector<int8_t> smaller_node_used_features =
      col_sampler_.GetByNode(tree, smaller_leaf_index);
  std::vector<int8_t> larger_node_used_features;
  if (larger_leaf_index >= 0) {
    larger_node_used_features = col_sampler_.GetByNode(tree, larger_leaf_index);
  }

  hist_t* smaller_hist =
      smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
  hist_t* larger_hist = nullptr;
  if (larger_leaf_histogram_array_ != nullptr) {
    larger_hist = larger_leaf_histogram_array_[0].RawData() - kHistOffset;
  }

  if (use_subtract && larger_hist != nullptr) {
    Common::FunctionTimer subtract_timer(
        "MetalSingleGPUTreeLearner::SubtractHistograms", global_timer);
    for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
      if (!is_feature_used[feature_index]) {
        continue;
      }
        larger_leaf_histogram_array_[feature_index].Subtract<false>(
            smaller_leaf_histogram_array_[feature_index]);
    }
  }
  log_stage("split/subtract");

  CHECK(best_split_finder_ != nullptr);
  best_split_finder_->FindBestSplitsForLeaf(
      smaller_hist,
      smaller_leaf,
      smaller_leaf_index,
      smaller_node_used_features,
      larger_hist,
      larger_leaf,
      larger_leaf_index,
      larger_leaf_index >= 0 ? &larger_node_used_features : nullptr);
  log_stage("split/gpu_search");

  auto recompute_leaf_on_cpu =
      [&](int leaf_index, LeafSplits* leaf_splits, FeatureHistogram* hist_array,
          const std::vector<int8_t>& node_used_features,
          double parent_output) {
        std::vector<SplitInfo> thread_best(share_state_->num_threads);
        OMP_INIT_EX();
#pragma omp parallel for schedule(static) num_threads(share_state_->num_threads)
        for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
          OMP_LOOP_EX_BEGIN();
          if (!is_feature_used[feature_index]) {
            continue;
          }
          const int tid = omp_get_thread_num();
          const int real_fidx = train_data_->RealFeatureIndex(feature_index);
          ComputeBestSplitForFeature(
              hist_array, feature_index, real_fidx,
              node_used_features[feature_index], leaf_splits->num_data_in_leaf(),
              leaf_splits, &thread_best[tid], parent_output);
          OMP_LOOP_EX_END();
        }
        OMP_THROW_EX();
        best_split_per_leaf_[leaf_index] =
            thread_best[ArrayArgs<SplitInfo>::ArgMax(thread_best)];
      };

  auto refine_leaf_split =
      [&](int leaf_index, LeafSplits* leaf_splits, FeatureHistogram* hist_array,
          const std::vector<int8_t>& node_used_features,
          double parent_output) {
        SplitInfo& split = best_split_per_leaf_[leaf_index];
        if (split.feature < 0 || split.gain <= kMinScore ||
            !node_used_features[split.feature]) {
          return;
        }

        SplitInfo refined;
        ComputeBestSplitForFeature(
            hist_array, split.feature, train_data_->RealFeatureIndex(split.feature),
            node_used_features[split.feature], leaf_splits->num_data_in_leaf(),
            leaf_splits, &refined, parent_output);

        if (refined.feature >= 0 && refined.gain > kMinScore &&
            refined.left_count > 0 && refined.right_count > 0) {
          split = refined;
        } else {
          recompute_leaf_on_cpu(leaf_index, leaf_splits, hist_array,
                                node_used_features, parent_output);
        }
      };

  if (smaller_leaf_index >= 0) {
    best_split_finder_->GetBestSplitForLeaf(
        smaller_leaf_index, &best_split_per_leaf_[smaller_leaf_index]);
    refine_leaf_split(smaller_leaf_index, smaller_leaf_splits_.get(),
                      smaller_leaf_histogram_array_,
                      smaller_node_used_features,
                      GetMetalParentOutput(tree, smaller_leaf));
    if (debug_logging) {
      const SplitInfo& split = best_split_per_leaf_[smaller_leaf_index];
      fprintf(stderr,
              "[Metal] leaf=%d feature=%d threshold=%u counts=%d/%d gain=%.8f default_left=%d\n",
              smaller_leaf_index, split.feature, split.threshold,
              static_cast<int>(split.left_count),
              static_cast<int>(split.right_count), split.gain,
              static_cast<int>(split.default_left));
    }
  }
  log_stage("split/refine_smaller");
  if (larger_leaf_index >= 0) {
    best_split_finder_->GetBestSplitForLeaf(
        larger_leaf_index, &best_split_per_leaf_[larger_leaf_index]);
    refine_leaf_split(larger_leaf_index, larger_leaf_splits_.get(),
                      larger_leaf_histogram_array_,
                      larger_node_used_features,
                      GetMetalParentOutput(tree, larger_leaf));
    if (debug_logging) {
      const SplitInfo& split = best_split_per_leaf_[larger_leaf_index];
      fprintf(stderr,
              "[Metal] leaf=%d feature=%d threshold=%u counts=%d/%d gain=%.8f default_left=%d\n",
              larger_leaf_index, split.feature, split.threshold,
              static_cast<int>(split.left_count),
              static_cast<int>(split.right_count), split.gain,
              static_cast<int>(split.default_left));
    }
  }
  log_stage("split/refine_larger");
}

data_size_t MetalSingleGPUTreeLearner::PartitionLeafOnGPU(
    int leaf,
    int inner_feature_index,
    uint32_t threshold,
    bool default_left) {
  CHECK(partition_pipeline_ != nullptr);
  const int group = train_data_->Feature2Group(inner_feature_index);
  CHECK(group >= 0);
  const MetalLeafSplitsStruct* leaf_state = FindActiveMetalLeafState(leaf);
  const data_size_t begin = data_partition_->leaf_begin(leaf);
  const data_size_t cnt = data_partition_->leaf_count(leaf);
  if (leaf_state != nullptr &&
      (leaf_state->data_indices_offset != begin ||
       leaf_state->num_data_in_leaf != cnt) &&
      std::getenv("LIGHTGBM_METAL_DEBUG") != nullptr) {
    fprintf(stderr,
            "[Metal] active leaf state mismatch for leaf=%d: state_offset=%d partition_offset=%d state_count=%d partition_count=%d\n",
            leaf, static_cast<int>(leaf_state->data_indices_offset),
            static_cast<int>(begin), static_cast<int>(leaf_state->num_data_in_leaf),
            static_cast<int>(cnt));
  }
  const data_size_t* leaf_indices = data_partition_->indices() + begin;

  @autoreleasepool {
    id<MTLBuffer> input_buf = (__bridge id<MTLBuffer>)data_indices_buffer_;
    std::memcpy([input_buf contents], leaf_indices, cnt * sizeof(data_size_t));

    id<MTLBuffer> count_buf = (__bridge id<MTLBuffer>)partition_counts_buffer_;
    std::memset([count_buf contents], 0, 2 * sizeof(uint32_t));

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metal_queue_;
    id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)partition_pipeline_;

    const BinMapper* bin_mapper = train_data_->FeatureBinMapper(inner_feature_index);
    const uint32_t most_freq_bin = bin_mapper->GetMostFreqBin();
    const uint32_t max_bin =
        static_cast<uint32_t>(train_data_->FeatureGroupNumBin(group) - 1);
    const uint32_t raw_threshold =
        threshold + (most_freq_bin == 0 ? 0u : 1u);
    const uint32_t raw_default_bin =
        bin_mapper->GetDefaultBin() + (most_freq_bin == 0 ? 0u : 1u);
    const MissingType missing_type = bin_mapper->missing_type();
    const int32_t split_default_to_left =
        (most_freq_bin <= threshold) ? 1 : 0;
    const int32_t split_missing_default_to_left =
        ((missing_type == MissingType::Zero ||
          missing_type == MissingType::NaN) && default_left) ? 1 : 0;
    const int32_t missing_is_zero = missing_type == MissingType::Zero ? 1 : 0;
    const int32_t missing_is_na = missing_type == MissingType::NaN ? 1 : 0;
    const int32_t mfb_is_zero =
        (missing_type == MissingType::Zero &&
         bin_mapper->GetDefaultBin() == most_freq_bin) ? 1 : 0;
    const int32_t mfb_is_na =
        (missing_type == MissingType::NaN && most_freq_bin > 0 &&
         most_freq_bin + 1 == max_bin) ? 1 : 0;
    const uint32_t num_data_in_leaf = static_cast<uint32_t>(cnt);

    [encoder setComputePipelineState:pso];
    [encoder setBuffer:(__bridge id<MTLBuffer>)bin_data_col_buffer_
                offset:static_cast<NSUInteger>(group) * static_cast<NSUInteger>(num_data_)
               atIndex:0];
    [encoder setBuffer:input_buf offset:0 atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)partition_output_buffer_ offset:0 atIndex:2];
    [encoder setBuffer:count_buf offset:0 atIndex:3];
    [encoder setBytes:&num_data_in_leaf length:sizeof(num_data_in_leaf) atIndex:4];
    [encoder setBytes:&raw_threshold length:sizeof(raw_threshold) atIndex:5];
    [encoder setBytes:&raw_default_bin length:sizeof(raw_default_bin) atIndex:6];
    [encoder setBytes:&max_bin length:sizeof(max_bin) atIndex:7];
    [encoder setBytes:&split_default_to_left
                length:sizeof(split_default_to_left) atIndex:8];
    [encoder setBytes:&split_missing_default_to_left
                length:sizeof(split_missing_default_to_left) atIndex:9];
    [encoder setBytes:&missing_is_zero length:sizeof(missing_is_zero) atIndex:10];
    [encoder setBytes:&missing_is_na length:sizeof(missing_is_na) atIndex:11];
    [encoder setBytes:&mfb_is_zero length:sizeof(mfb_is_zero) atIndex:12];
    [encoder setBytes:&mfb_is_na length:sizeof(mfb_is_na) atIndex:13];

    const NSUInteger tg =
        std::min(256u, static_cast<uint32_t>([pso maxTotalThreadsPerThreadgroup]));
    [encoder dispatchThreadgroups:MTLSizeMake((num_data_in_leaf + tg - 1) / tg, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
    METAL_CHECK([cmd_buf status] != MTLCommandBufferStatusError,
                [[[cmd_buf error] localizedDescription] UTF8String]);

    const uint32_t* counts =
        reinterpret_cast<const uint32_t*>([count_buf contents]);
    const uint32_t left_count = counts[0];
    const uint32_t right_count = counts[1];
    CHECK_EQ(static_cast<uint32_t>(cnt), left_count + right_count);

    std::memcpy(data_partition_->mutable_indices() + begin,
                [(__bridge id<MTLBuffer>)partition_output_buffer_ contents],
                cnt * sizeof(data_size_t));
    return static_cast<data_size_t>(left_count);
  }
}

void MetalSingleGPUTreeLearner::Split(
    Tree* tree,
    int best_leaf,
    int* left_leaf,
    int* right_leaf) {
  Common::FunctionTimer fun_timer("MetalSingleGPUTreeLearner::Split", global_timer);
  const bool enable_gpu_partition =
      std::getenv("LIGHTGBM_METAL_ENABLE_GPU_PARTITION") != nullptr &&
      std::getenv("LIGHTGBM_METAL_DISABLE_GPU_PARTITION") == nullptr;
  if (!enable_gpu_partition) {
    SplitInner(tree, best_leaf, left_leaf, right_leaf, true);
    SyncMetalActiveLeafState();
    return;
  }
  const bool stage_timing =
      std::getenv("LIGHTGBM_METAL_STAGE_TIMING") != nullptr;
  auto stage_start = std::chrono::steady_clock::now();
  auto log_stage = [&](const char* name) {
    if (!stage_timing) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(
        now - stage_start).count();
    Log::Info("[MetalTiming] %s took %.3f ms", name, ms);
    stage_start = now;
  };
  if (std::getenv("LIGHTGBM_METAL_DEBUG") != nullptr) {
    const SplitInfo& split = best_split_per_leaf_[best_leaf];
    fprintf(stderr,
            "[Metal] split leaf=%d feature=%d threshold=%u counts=%d/%d gain=%.8f default_left=%d\n",
            best_leaf, split.feature, split.threshold,
            static_cast<int>(split.left_count),
            static_cast<int>(split.right_count), split.gain,
            static_cast<int>(split.default_left));
  }
  SplitInfo& best_split_info = best_split_per_leaf_[best_leaf];
  const int inner_feature_index =
      train_data_->InnerFeatureIndex(best_split_info.feature);
  if (cegb_ != nullptr) {
    cegb_->UpdateLeafBestSplits(tree, best_leaf, &best_split_info,
                                &best_split_per_leaf_);
  }
  *left_leaf = best_leaf;
  const int next_leaf_id = tree->NextLeafId();

  constraints_->BeforeSplit(best_leaf, next_leaf_id,
                            best_split_info.monotone_type);

  const bool is_numerical_split =
      train_data_->FeatureBinMapper(inner_feature_index)->bin_type() ==
      BinType::NumericalBin;
  if (!is_numerical_split) {
    Log::Fatal("Metal tree learner only supports numerical splits.");
  }

  const data_size_t left_count = PartitionLeafOnGPU(
      best_leaf, inner_feature_index, best_split_info.threshold,
      best_split_info.default_left);
  const data_size_t total_count = data_partition_->leaf_count(best_leaf);
  const data_size_t right_count = total_count - left_count;
  data_partition_->ApplyExternalSplit(best_leaf, next_leaf_id, left_count);
  best_split_info.left_count = left_count;
  best_split_info.right_count = right_count;
  log_stage("partition/gpu");

  const auto threshold_double =
      train_data_->RealThreshold(inner_feature_index, best_split_info.threshold);
  *right_leaf = tree->Split(
      best_leaf, inner_feature_index, best_split_info.feature,
      best_split_info.threshold, threshold_double,
      static_cast<double>(best_split_info.left_output),
      static_cast<double>(best_split_info.right_output),
      static_cast<data_size_t>(best_split_info.left_count),
      static_cast<data_size_t>(best_split_info.right_count),
      static_cast<double>(best_split_info.left_sum_hessian),
      static_cast<double>(best_split_info.right_sum_hessian),
      static_cast<float>(best_split_info.gain + config_->min_gain_to_split),
      train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
      best_split_info.default_left);
  log_stage("partition/tree_split");
  UpdateMetalLeafState(*left_leaf, best_split_info.left_sum_gradient,
                       best_split_info.left_sum_hessian,
                       best_split_info.left_count, best_split_info.left_output);
  UpdateMetalLeafState(*right_leaf, best_split_info.right_sum_gradient,
                       best_split_info.right_sum_hessian,
                       best_split_info.right_count, best_split_info.right_output);
  log_stage("partition/update_leaf_state");

#ifdef DEBUG
  CHECK(*right_leaf == next_leaf_id);
#endif

  if (best_split_info.left_count < best_split_info.right_count) {
    CHECK_GT(best_split_info.left_count, 0);
    smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                               best_split_info.left_sum_gradient,
                               best_split_info.left_sum_hessian,
                               best_split_info.left_output);
    larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                              best_split_info.right_sum_gradient,
                              best_split_info.right_sum_hessian,
                              best_split_info.right_output);
  } else {
    CHECK_GT(best_split_info.right_count, 0);
    smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                               best_split_info.right_sum_gradient,
                               best_split_info.right_sum_hessian,
                               best_split_info.right_output);
    larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                              best_split_info.left_sum_gradient,
                              best_split_info.left_sum_hessian,
                              best_split_info.left_output);
  }
  log_stage("partition/leaf_init");

#ifdef DEBUG
  CheckSplit(best_split_info, *left_leaf, *right_leaf);
#endif

  auto leaves_need_update = constraints_->Update(
      true, *left_leaf, *right_leaf,
      best_split_info.monotone_type, best_split_info.right_output,
      best_split_info.left_output, inner_feature_index, best_split_info,
      best_split_per_leaf_);
  for (auto leaf : leaves_need_update) {
    RecomputeBestSplitForLeaf(tree, leaf, &best_split_per_leaf_[leaf]);
  }
  log_stage("partition/constraints");
}

// ============================================================================
// AllocateMetalBuffers / ResetTrainingData
// ============================================================================

void MetalSingleGPUTreeLearner::AllocateMetalBuffers() {
  if (!num_dense_feature_groups_) {
    Log::Warning("Metal GPU acceleration disabled — no dense features found");
    return;
  }
  @autoreleasepool {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)metal_device_;
    auto alloc = [&](void*& p, size_t bytes) {
      if (p) CFRelease(p);
      p = (__bridge_retained void*)
          [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    };
    alloc(gradients_buffer_, num_data_ * sizeof(score_t));
    alloc(hessians_buffer_, num_data_ * sizeof(score_t));
    alloc(data_indices_buffer_, num_data_ * sizeof(data_size_t));
    alloc(partition_output_buffer_, num_data_ * sizeof(data_size_t));
    alloc(partition_counts_buffer_, 2 * sizeof(uint32_t));
  }
}

void MetalSingleGPUTreeLearner::ResetTrainingData(
    const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  ValidateTrainingScope(train_data_);
  num_feature_groups_ = train_data_->num_feature_groups();
  bin_data_packed_ = false;
  ResetMetalLeafStateTable();
  if (best_split_finder_ != nullptr) {
    best_split_finder_->ResetTrainingData(
        train_data_, share_state_->feature_hist_offsets());
    best_split_finder_->ResetConfig(config_);
  }
  AllocateMetalBuffers();
}

}  // namespace LightGBM

#endif
