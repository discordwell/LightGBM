/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef LGBM_USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_histogram_constructor.hpp"

#include <LightGBM/feature_group.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace LightGBM {

// ---------------------------------------------------------------------------
//  Construction / destruction
// ---------------------------------------------------------------------------

MetalHistogramConstructor::MetalHistogramConstructor(
    const Dataset* train_data,
    int num_leaves,
    int num_threads,
    const std::vector<uint32_t>& feature_hist_offsets,
    int min_data_in_leaf,
    double min_sum_hessian_in_leaf)
    : num_data_(train_data->num_data()),
      num_features_(train_data->num_features()),
      num_leaves_(num_leaves),
      num_threads_(num_threads),
      min_data_in_leaf_(min_data_in_leaf),
      min_sum_hessian_in_leaf_(min_sum_hessian_in_leaf),
      gradients_(nullptr),
      hessians_(nullptr),
      histogram_pso_(nullptr),
      subtract_pso_(nullptr),
      row_data_bit_type_(8) {
  InitFeatureMetaInfo(train_data, feature_hist_offsets);
}

MetalHistogramConstructor::~MetalHistogramConstructor() {}

// ---------------------------------------------------------------------------
//  Feature metadata
// ---------------------------------------------------------------------------

void MetalHistogramConstructor::InitFeatureMetaInfo(
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets) {
  need_fix_histogram_features_.clear();
  need_fix_histogram_features_num_bin_aligned_.clear();
  feature_num_bins_.clear();
  feature_most_freq_bins_.clear();

  for (int feature_index = 0; feature_index < train_data->num_features();
       ++feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    const uint32_t most_freq_bin = bin_mapper->GetMostFreqBin();
    if (most_freq_bin != 0) {
      need_fix_histogram_features_.emplace_back(feature_index);
      uint32_t num_bin_ref =
          static_cast<uint32_t>(bin_mapper->num_bin()) - 1;
      uint32_t num_bin_aligned = 1;
      while (num_bin_ref > 0) {
        num_bin_aligned <<= 1;
        num_bin_ref >>= 1;
      }
      need_fix_histogram_features_num_bin_aligned_.emplace_back(
          num_bin_aligned);
    }
    feature_num_bins_.emplace_back(
        static_cast<uint32_t>(bin_mapper->num_bin()));
    feature_most_freq_bins_.emplace_back(most_freq_bin);
  }

  feature_hist_offsets_.clear();
  for (size_t i = 0; i < feature_hist_offsets.size(); ++i) {
    feature_hist_offsets_.emplace_back(feature_hist_offsets[i]);
  }
  if (feature_hist_offsets.empty()) {
    num_total_bin_ = 0;
  } else {
    num_total_bin_ = static_cast<int>(feature_hist_offsets.back());
  }
}

// ---------------------------------------------------------------------------
//  Row data packing for GPU
// ---------------------------------------------------------------------------

void MetalHistogramConstructor::InitRowData(
    const Dataset* train_data,
    TrainingShareStates* share_state) {
  fprintf(stderr, "[Metal] InitRowData: start, num_features=%d num_data=%d\n", num_features_, num_data_);
  uint32_t max_bin = 0;
  for (int f = 0; f < num_features_; ++f) {
    const uint32_t nb = feature_num_bins_[f];
    if (nb > max_bin) {
      max_bin = nb;
    }
  }
  if (max_bin <= 256) {
    row_data_bit_type_ = 8;
  } else {
    row_data_bit_type_ = 16;
  }

  // Pack row-wise bin data into a contiguous MetalBuffer.
  // Layout: row-major, each row has num_features_ entries, each entry is
  // either uint8 (bit_type=8) or uint16 (bit_type=16).
  const size_t bytes_per_row = (row_data_bit_type_ == 8)
      ? static_cast<size_t>(num_features_)
      : static_cast<size_t>(num_features_) * 2;
  const size_t total_bytes = bytes_per_row * static_cast<size_t>(num_data_);
  row_bin_data_.Resize(total_bytes);
  uint8_t* dst = row_bin_data_.data();
  std::memset(dst, 0, total_bytes);

  // Fill row data from the dataset's feature iterators.
  // Each BinIterator provides bin values per data point for a feature.
  // The returned pointers are owned by the Dataset; do not delete them.
  fprintf(stderr, "[Metal] InitRowData: allocating %zu bytes, bit_type=%d\n", total_bytes, row_data_bit_type_);
  std::vector<BinIterator*> iterators(num_features_);
  for (int f = 0; f < num_features_; ++f) {
    iterators[f] = train_data->FeatureIterator(f);
    iterators[f]->Reset(0);
  }
  fprintf(stderr, "[Metal] InitRowData: iterators ready, packing rows...\n");

  if (row_data_bit_type_ == 8) {
    #pragma omp parallel for num_threads(num_threads_) schedule(static)
    for (data_size_t row = 0; row < num_data_; ++row) {
      uint8_t* row_dst = dst + static_cast<size_t>(row) * num_features_;
      for (int f = 0; f < num_features_; ++f) {
        row_dst[f] = static_cast<uint8_t>(iterators[f]->RawGet(row));
      }
    }
  } else {
    uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst);
    #pragma omp parallel for num_threads(num_threads_) schedule(static)
    for (data_size_t row = 0; row < num_data_; ++row) {
      uint16_t* row_dst = dst16 + static_cast<size_t>(row) * num_features_;
      for (int f = 0; f < num_features_; ++f) {
        row_dst[f] = static_cast<uint16_t>(iterators[f]->RawGet(row));
      }
    }
  }
  fprintf(stderr, "[Metal] InitRowData: done\n");
}

// ---------------------------------------------------------------------------
//  Init
// ---------------------------------------------------------------------------

void MetalHistogramConstructor::Init(
    const Dataset* train_data,
    TrainingShareStates* share_state) {
  fprintf(stderr, "[Metal] HistogramConstructor::Init: start, bins=%d leaves=%d features=%d\n",
          num_total_bin_, num_leaves_, num_features_);
  const size_t hist_size =
      static_cast<size_t>(num_total_bin_) * 2 *
      static_cast<size_t>(num_leaves_);
  hist_buf_.Resize(hist_size);
  std::memset(hist_buf_.data(), 0, hist_size * sizeof(hist_t));
  // Separate float buffer for GPU output (kernel writes float, host uses double)
  gpu_hist_float_.Resize(hist_size);
  std::memset(gpu_hist_float_.data(), 0, hist_size * sizeof(float));
  fprintf(stderr, "[Metal] HistogramConstructor::Init: hist_buf allocated (%zu entries)\n", hist_size);

  feature_num_bins_buf_.Resize(feature_num_bins_.size());
  std::memcpy(feature_num_bins_buf_.data(), feature_num_bins_.data(),
              feature_num_bins_.size() * sizeof(uint32_t));

  feature_hist_offsets_buf_.Resize(feature_hist_offsets_.size());
  std::memcpy(feature_hist_offsets_buf_.data(), feature_hist_offsets_.data(),
              feature_hist_offsets_.size() * sizeof(uint32_t));

  feature_most_freq_bins_buf_.Resize(feature_most_freq_bins_.size());
  std::memcpy(feature_most_freq_bins_buf_.data(),
              feature_most_freq_bins_.data(),
              feature_most_freq_bins_.size() * sizeof(uint32_t));
  fprintf(stderr, "[Metal] HistogramConstructor::Init: metadata copied, calling InitRowData\n");

  InitRowData(train_data, share_state);
  fprintf(stderr, "[Metal] HistogramConstructor::Init: InitRowData done\n");

  histogram_pso_ = MetalDevice::GetPipeline("histogram_dense");
  fprintf(stderr, "[Metal] HistogramConstructor::Init: histogram_dense pipeline=%p\n", histogram_pso_);
  subtract_pso_ = MetalDevice::GetPipeline("histogram_subtract");
  fprintf(stderr, "[Metal] HistogramConstructor::Init: done\n");
}

// ---------------------------------------------------------------------------
//  BeforeTrain
// ---------------------------------------------------------------------------

void MetalHistogramConstructor::BeforeTrain(const score_t* gradients,
                                            const score_t* hessians) {
  gradients_ = gradients;
  hessians_ = hessians;
  // Zero the histogram buffer for the new iteration.
  std::memset(hist_buf_.data(), 0,
              hist_buf_.size() * sizeof(hist_t));
}

// ---------------------------------------------------------------------------
//  ConstructHistogramForLeaf — dispatch Metal histogram kernel
// ---------------------------------------------------------------------------

void MetalHistogramConstructor::ConstructHistogramForLeaf(
    const MetalLeafSplitsStruct* smaller_leaf,
    const MetalLeafSplitsStruct* larger_leaf,
    data_size_t num_data_in_smaller_leaf,
    data_size_t num_data_in_larger_leaf,
    double sum_hessians_in_smaller_leaf,
    double sum_hessians_in_larger_leaf) {
  fprintf(stderr, "[Metal] ConstructHistogram: smaller=%d(%d rows) larger=%d(%d rows)\n",
          smaller_leaf->leaf_index, num_data_in_smaller_leaf,
          larger_leaf ? larger_leaf->leaf_index : -1, num_data_in_larger_leaf);
  // Skip if both leaves fail minimum constraints.
  if ((num_data_in_smaller_leaf <= min_data_in_leaf_ ||
       sum_hessians_in_smaller_leaf <= min_sum_hessian_in_leaf_) &&
      (num_data_in_larger_leaf <= min_data_in_leaf_ ||
       sum_hessians_in_larger_leaf <= min_sum_hessian_in_leaf_)) {
    fprintf(stderr, "[Metal] ConstructHistogram: skipped (min constraints)\n");
    return;
  }

  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)MetalDevice::GetQueue();
  id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
  fprintf(stderr, "[Metal] ConstructHistogram: command buffer created\n");

  id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
  id<MTLComputePipelineState> pso =
      (__bridge id<MTLComputePipelineState>)histogram_pso_;
  [encoder setComputePipelineState:pso];

  // Buffer 0-1: gradients and hessians.
  // Create Metal buffers wrapping the host pointers. We use
  // newBufferWithBytes which copies into GPU-visible shared memory.
  // On Apple Silicon this is fast (~unified memory DMA).
  id<MTLDevice> device = (__bridge id<MTLDevice>)MetalDevice::GetDevice();
  const NSUInteger grad_hess_bytes =
      static_cast<NSUInteger>(num_data_) * sizeof(score_t);
  id<MTLBuffer> grad_buf = [device
      newBufferWithBytes:gradients_
                  length:grad_hess_bytes
                 options:MTLResourceStorageModeShared];
  id<MTLBuffer> hess_buf = [device
      newBufferWithBytes:hessians_
                  length:grad_hess_bytes
                 options:MTLResourceStorageModeShared];

  [encoder setBuffer:grad_buf offset:0 atIndex:0];
  [encoder setBuffer:hess_buf offset:0 atIndex:1];

  // Buffer 2: row bin data
  [encoder setBuffer:(__bridge id<MTLBuffer>)row_bin_data_.GetMTLBuffer()
              offset:0
             atIndex:2];

  // Buffer 3: feature hist offsets
  [encoder setBuffer:(__bridge id<MTLBuffer>)feature_hist_offsets_buf_.GetMTLBuffer()
              offset:0
             atIndex:3];

  // Buffer 4: feature most frequent bins
  [encoder setBuffer:(__bridge id<MTLBuffer>)feature_most_freq_bins_buf_.GetMTLBuffer()
              offset:0
             atIndex:4];

  // Buffer 5: GPU float histogram output — offset to the smaller leaf's region.
  // The kernel writes float; we convert to hist_t (double) after completion.
  const int64_t smaller_hist_offset = smaller_leaf->hist_offset;
  // Zero the GPU output region for this leaf
  std::memset(gpu_hist_float_.data() + smaller_hist_offset,
              0, num_total_bin_ * 2 * sizeof(float));
  [encoder setBuffer:(__bridge id<MTLBuffer>)gpu_hist_float_.GetMTLBuffer()
              offset:static_cast<NSUInteger>(smaller_hist_offset) * sizeof(float)
             atIndex:5];

  // Buffer 6: constants struct
  struct HistogramParams {
    uint32_t num_data;
    uint32_t num_features;
    uint32_t num_total_bin;
    uint32_t bit_type;
  };
  HistogramParams params;
  params.num_data = static_cast<uint32_t>(num_data_in_smaller_leaf);
  params.num_features = static_cast<uint32_t>(num_features_);
  params.num_total_bin = static_cast<uint32_t>(num_total_bin_);
  params.bit_type = static_cast<uint32_t>(row_data_bit_type_);
  [encoder setBytes:&params length:sizeof(params) atIndex:6];

  // Buffer 7: data indices for the smaller leaf (may be full range if root)
  const data_size_t data_offset = smaller_leaf->data_indices_offset;
  [encoder setBytes:&data_offset length:sizeof(data_offset) atIndex:7];

  // Buffer 8: feature num bins
  [encoder setBuffer:(__bridge id<MTLBuffer>)feature_num_bins_buf_.GetMTLBuffer()
              offset:0
             atIndex:8];

  // Dispatch: one threadgroup per feature, threads cover data points.
  NSUInteger threadgroup_size = std::min(
      static_cast<NSUInteger>(256),
      [pso maxTotalThreadsPerThreadgroup]);
  NSUInteger num_threadgroups = static_cast<NSUInteger>(num_features_);
  [encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];

  [encoder endEncoding];
  fprintf(stderr, "[Metal] ConstructHistogram: dispatched %lu groups × %lu threads, committing...\n",
          (unsigned long)num_threadgroups, (unsigned long)threadgroup_size);
  [cmdBuf commit];
  fprintf(stderr, "[Metal] ConstructHistogram: committed, waiting...\n");
  [cmdBuf waitUntilCompleted];
  if ([cmdBuf status] == MTLCommandBufferStatusError) {
    fprintf(stderr, "[Metal] ConstructHistogram: GPU ERROR: %s\n",
            [[[cmdBuf error] localizedDescription] UTF8String]);
  }
  fprintf(stderr, "[Metal] ConstructHistogram: GPU done\n");

  // Convert GPU float histogram → host double histogram.
  {
    const float* src = gpu_hist_float_.data() + smaller_hist_offset;
    hist_t* dst = hist_buf_.data() + smaller_hist_offset;
    const size_t n = static_cast<size_t>(num_total_bin_) * 2;
    for (size_t j = 0; j < n; ++j) {
      dst[j] = static_cast<hist_t>(src[j]);
    }
  }

  // Debug: dump histogram for first feature
  {
    hist_t* h = hist_buf_.data() + smaller_hist_offset;
    double gsum = 0, hsum = 0;
    int nonzero = 0;
    uint32_t nfbins = feature_num_bins_[0];
    for (uint32_t b = 0; b < nfbins; ++b) {
      double g = h[feature_hist_offsets_[0] * 2 + b * 2];
      double hs = h[feature_hist_offsets_[0] * 2 + b * 2 + 1];
      gsum += g;
      hsum += hs;
      if (g != 0 || hs != 0) nonzero++;
    }
    fprintf(stderr, "[Metal] Histogram feature 0: %d/%d non-zero bins, gsum=%.4f hsum=%.4f\n",
            nonzero, nfbins, gsum, hsum);
  }

  // Fix histogram for features with most_freq_bin != 0.
  // The GPU kernel skips the most-frequent-bin slot; we reconstruct it
  // from the leaf totals by subtracting the sum of all other bins.
  if (!need_fix_histogram_features_.empty()) {
    hist_t* hist = hist_buf_.data() + smaller_hist_offset;
    const double total_grad = smaller_leaf->sum_of_gradients;
    const double total_hess = smaller_leaf->sum_of_hessians;
    for (size_t k = 0; k < need_fix_histogram_features_.size(); ++k) {
      const int fidx = need_fix_histogram_features_[k];
      const uint32_t offset = feature_hist_offsets_[fidx];
      const uint32_t num_bin = feature_num_bins_[fidx];
      const uint32_t mfb = feature_most_freq_bins_[fidx];
      double sum_grad = 0.0;
      double sum_hess = 0.0;
      for (uint32_t b = 0; b < num_bin; ++b) {
        if (b == mfb) continue;
        sum_grad += hist[(offset + b) * 2];
        sum_hess += hist[(offset + b) * 2 + 1];
      }
      hist[(offset + mfb) * 2] = total_grad - sum_grad;
      hist[(offset + mfb) * 2 + 1] = total_hess - sum_hess;
    }
  }
}

// ---------------------------------------------------------------------------
//  SubtractHistogramForLeaf — dispatch subtraction kernel
// ---------------------------------------------------------------------------

void MetalHistogramConstructor::SubtractHistogramForLeaf(
    const MetalLeafSplitsStruct* smaller_leaf,
    const MetalLeafSplitsStruct* larger_leaf) {
  if (larger_leaf->leaf_index < 0) {
    return;
  }

  // parent_hist = smaller_hist, since the parent was split into
  // smaller + larger; we compute larger = parent - smaller.
  // The parent histogram is at the smaller leaf's location before the
  // split overwrote it — but in our design the histogram_constructor
  // builds the smaller-leaf histogram fresh each iteration, and the
  // parent histogram was at min(smaller, larger) index.
  // For subtraction: larger_hist = parent_hist - smaller_hist.
  const int parent_leaf = std::min(smaller_leaf->leaf_index,
                                   larger_leaf->leaf_index);
  const int64_t parent_hist_offset =
      static_cast<int64_t>(parent_leaf) *
      static_cast<int64_t>(num_total_bin_) * 2;
  const int64_t smaller_hist_offset = smaller_leaf->hist_offset;
  const int64_t larger_hist_offset = larger_leaf->hist_offset;

  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)MetalDevice::GetQueue();
  id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

  id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
  id<MTLComputePipelineState> pso =
      (__bridge id<MTLComputePipelineState>)subtract_pso_;
  [encoder setComputePipelineState:pso];

  // Buffer 0: full histogram buffer
  [encoder setBuffer:(__bridge id<MTLBuffer>)hist_buf_.GetMTLBuffer()
              offset:0
             atIndex:0];

  // Buffer 1: offsets struct
  struct SubtractParams {
    int64_t parent_offset;
    int64_t smaller_offset;
    int64_t larger_offset;
    uint32_t num_bins_x2;
  };
  SubtractParams sp;
  sp.parent_offset = parent_hist_offset;
  sp.smaller_offset = smaller_hist_offset;
  sp.larger_offset = larger_hist_offset;
  sp.num_bins_x2 = static_cast<uint32_t>(num_total_bin_) * 2;
  [encoder setBytes:&sp length:sizeof(sp) atIndex:1];

  NSUInteger threadgroup_size = std::min(
      static_cast<NSUInteger>(1024),
      [pso maxTotalThreadsPerThreadgroup]);
  NSUInteger total_elements = static_cast<NSUInteger>(num_total_bin_) * 2;
  NSUInteger num_threadgroups =
      (total_elements + threadgroup_size - 1) / threadgroup_size;
  [encoder dispatchThreadgroups:MTLSizeMake(num_threadgroups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];

  [encoder endEncoding];
  [cmdBuf commit];
  [cmdBuf waitUntilCompleted];
}

// ---------------------------------------------------------------------------
//  ResetTrainingData
// ---------------------------------------------------------------------------

void MetalHistogramConstructor::ResetTrainingData(
    const Dataset* train_data,
    TrainingShareStates* share_states) {
  num_data_ = train_data->num_data();
  num_features_ = train_data->num_features();
  InitFeatureMetaInfo(train_data, share_states->feature_hist_offsets());

  const size_t hist_size =
      static_cast<size_t>(num_total_bin_) * 2 *
      static_cast<size_t>(num_leaves_);
  hist_buf_.Resize(hist_size);
  std::memset(hist_buf_.data(), 0, hist_size * sizeof(hist_t));

  feature_num_bins_buf_.Resize(feature_num_bins_.size());
  std::memcpy(feature_num_bins_buf_.data(), feature_num_bins_.data(),
              feature_num_bins_.size() * sizeof(uint32_t));

  feature_hist_offsets_buf_.Resize(feature_hist_offsets_.size());
  std::memcpy(feature_hist_offsets_buf_.data(), feature_hist_offsets_.data(),
              feature_hist_offsets_.size() * sizeof(uint32_t));

  feature_most_freq_bins_buf_.Resize(feature_most_freq_bins_.size());
  std::memcpy(feature_most_freq_bins_buf_.data(),
              feature_most_freq_bins_.data(),
              feature_most_freq_bins_.size() * sizeof(uint32_t));

  InitRowData(train_data, share_states);
}

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
