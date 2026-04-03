/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef LGBM_USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_best_split_finder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace LightGBM {

namespace {

double ThresholdL1(double s, double l1) {
  const double reg_s = std::fmax(0.0, std::fabs(s) - l1);
  return (s >= 0.0) ? reg_s : -reg_s;
}

double CalcLeafGain(double sum_grad, double sum_hess, double l1, double l2) {
  if (l1 > 0.0) {
    const double sg = ThresholdL1(sum_grad, l1);
    return (sg * sg) / (sum_hess + l2);
  }
  return (sum_grad * sum_grad) / (sum_hess + l2);
}

bool IsBetterResult(const MetalSplitResult& candidate,
                    const MetalSplitResult& best) {
  if (!candidate.found) {
    return false;
  }
  if (!best.found) {
    return true;
  }
  if (candidate.gain != best.gain) {
    return candidate.gain > best.gain;
  }
  const int candidate_feature =
      candidate.feature >= 0 ? candidate.feature : std::numeric_limits<int>::max();
  const int best_feature =
      best.feature >= 0 ? best.feature : std::numeric_limits<int>::max();
  return candidate_feature < best_feature;
}

}  // namespace

MetalBestSplitFinder::MetalBestSplitFinder(
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets,
    const Config* config)
    : num_features_(train_data->num_features()),
      num_leaves_(config->num_leaves),
      num_total_bin_(feature_hist_offsets.empty()
                         ? 0
                         : static_cast<int>(feature_hist_offsets.back())),
      feature_hist_offsets_(feature_hist_offsets),
      lambda_l1_(config->lambda_l1),
      lambda_l2_(config->lambda_l2),
      min_data_in_leaf_(config->min_data_in_leaf),
      min_sum_hessian_in_leaf_(config->min_sum_hessian_in_leaf),
      min_gain_to_split_(config->min_gain_to_split),
      num_tasks_(0),
      find_best_split_pso_(nullptr) {
  InitFeatureMetaInfo(train_data);
}

MetalBestSplitFinder::~MetalBestSplitFinder() {}

void MetalBestSplitFinder::InitFeatureMetaInfo(const Dataset* train_data) {
  feature_missing_type_.resize(num_features_);
  feature_mfb_offsets_.resize(num_features_);
  feature_default_bins_.resize(num_features_);
  feature_num_bins_.resize(num_features_);

  for (int f = 0; f < num_features_; ++f) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(f);
    if (bin_mapper->bin_type() != BinType::NumericalBin) {
      Log::Fatal("Metal tree learner only supports dense numerical features.");
    }
    feature_missing_type_[f] = bin_mapper->missing_type();
    feature_mfb_offsets_[f] =
        static_cast<uint32_t>(bin_mapper->GetMostFreqBin() == 0);
    feature_default_bins_[f] = bin_mapper->GetDefaultBin();
    feature_num_bins_[f] = static_cast<uint32_t>(bin_mapper->num_bin());
  }
}

void MetalBestSplitFinder::InitTasks() {
  num_tasks_ = 0;
  for (int f = 0; f < num_features_; ++f) {
    const uint32_t num_bin = feature_num_bins_[f];
    const MissingType mt = feature_missing_type_[f];
    if (num_bin > 2 && mt != MissingType::None) {
      num_tasks_ += 2;
    } else {
      ++num_tasks_;
    }
  }

  split_find_tasks_.resize(num_tasks_);
  int cur = 0;
  for (int f = 0; f < num_features_; ++f) {
    const uint32_t num_bin = feature_num_bins_[f];
    const MissingType mt = feature_missing_type_[f];
    if (num_bin > 2 && mt != MissingType::None) {
      MetalSplitFindTask& t0 = split_find_tasks_[cur++];
      t0.inner_feature_index = f;
      t0.reverse = 0;
      t0.skip_default_bin = (mt == MissingType::Zero) ? 1 : 0;
      t0.na_as_missing = (mt == MissingType::NaN) ? 1 : 0;
      t0.assume_out_default_left = 0;
      t0.hist_offset = feature_hist_offsets_[f];
      t0.mfb_offset = feature_mfb_offsets_[f];
      t0.default_bin = feature_default_bins_[f];
      t0.num_bin = num_bin;

      MetalSplitFindTask& t1 = split_find_tasks_[cur++];
      t1.inner_feature_index = f;
      t1.reverse = 1;
      t1.skip_default_bin = (mt == MissingType::Zero) ? 1 : 0;
      t1.na_as_missing = (mt == MissingType::NaN) ? 1 : 0;
      t1.assume_out_default_left = 1;
      t1.hist_offset = feature_hist_offsets_[f];
      t1.mfb_offset = feature_mfb_offsets_[f];
      t1.default_bin = feature_default_bins_[f];
      t1.num_bin = num_bin;
    } else {
      MetalSplitFindTask& t = split_find_tasks_[cur++];
      t.inner_feature_index = f;
      t.reverse = 1;
      t.skip_default_bin = 0;
      t.na_as_missing = 0;
      t.assume_out_default_left = (mt != MissingType::NaN) ? 1 : 0;
      t.hist_offset = feature_hist_offsets_[f];
      t.mfb_offset = feature_mfb_offsets_[f];
      t.default_bin = feature_default_bins_[f];
      t.num_bin = num_bin;
    }
  }

  tasks_buf_.Resize(static_cast<size_t>(num_tasks_));
  std::memcpy(tasks_buf_.data(), split_find_tasks_.data(),
              static_cast<size_t>(num_tasks_) * sizeof(MetalSplitFindTask));
}

void MetalBestSplitFinder::Init() {
  InitTasks();
  is_feature_used_buf_.Resize(static_cast<size_t>(num_features_));
  per_task_result_buf_.Resize(static_cast<size_t>(num_tasks_));
  per_leaf_best_buf_.Resize(static_cast<size_t>(num_leaves_));
  histogram_input_buf_.Resize(static_cast<size_t>(num_total_bin_) * 2 * 2);
  find_best_split_pso_ = MetalDevice::GetPipeline("find_best_split_numeric");
}

void MetalBestSplitFinder::BeforeTrain(
    const std::vector<int8_t>& is_feature_used_bytree) {
  std::memcpy(is_feature_used_buf_.data(), is_feature_used_bytree.data(),
              is_feature_used_bytree.size() * sizeof(int8_t));
  for (int i = 0; i < num_leaves_; ++i) {
    ClearLeafBest(i);
  }
}

void MetalBestSplitFinder::ClearLeafBest(int leaf_index) {
  if (leaf_index < 0 || leaf_index >= num_leaves_) {
    return;
  }
  MetalSplitResult& best = per_leaf_best_buf_.data()[leaf_index];
  best.gain = -std::numeric_limits<float>::infinity();
  best.feature = -1;
  best.threshold = 0;
  best.default_left = 1;
  best.left_sum_gradient = 0.0f;
  best.left_sum_hessian = 0.0f;
  best.left_count = 0;
  best.right_sum_gradient = 0.0f;
  best.right_sum_hessian = 0.0f;
  best.right_count = 0;
  best.left_value = 0.0f;
  best.right_value = 0.0f;
  best.found = 0;
}

void MetalBestSplitFinder::UploadHistogram(const hist_t* src_hist, size_t slot) {
  const size_t hist_items = static_cast<size_t>(num_total_bin_) * 2;
  float* dst = histogram_input_buf_.data() + slot * hist_items;
  for (size_t i = 0; i < hist_items; ++i) {
    dst[i] = static_cast<float>(src_hist[i]);
  }
}

void MetalBestSplitFinder::DispatchSplitKernel(
    const LeafSplits* leaf_splits,
    int leaf_index,
    size_t hist_slot,
    const std::vector<int8_t>& node_feature_mask) {
  if (leaf_index < 0 || leaf_splits == nullptr ||
      leaf_splits->num_data_in_leaf() <= min_data_in_leaf_ ||
      leaf_splits->sum_hessians() <= min_sum_hessian_in_leaf_) {
    ClearLeafBest(leaf_index);
    return;
  }

  const float total_grad = static_cast<float>(leaf_splits->sum_gradients());
  const float total_hess = static_cast<float>(leaf_splits->sum_hessians());
  const uint32_t total_count =
      static_cast<uint32_t>(leaf_splits->num_data_in_leaf());
  const float parent_gain = static_cast<float>(
      CalcLeafGain(leaf_splits->sum_gradients(), leaf_splits->sum_hessians(),
                   lambda_l1_, lambda_l2_));
  const float lambda_l1 = static_cast<float>(lambda_l1_);
  const float lambda_l2 = static_cast<float>(lambda_l2_);
  const float min_gain_to_split = static_cast<float>(min_gain_to_split_);
  const int32_t min_data_in_leaf = min_data_in_leaf_;
  const float min_sum_hessian_in_leaf =
      static_cast<float>(min_sum_hessian_in_leaf_);
  const uint32_t num_tasks = static_cast<uint32_t>(num_tasks_);
  const size_t hist_offset =
      hist_slot * static_cast<size_t>(num_total_bin_) * 2 * sizeof(float);

  @autoreleasepool {
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)MetalDevice::GetQueue();
    id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    id<MTLComputePipelineState> pso =
        (__bridge id<MTLComputePipelineState>)find_best_split_pso_;

    [encoder setComputePipelineState:pso];
    [encoder setBuffer:(__bridge id<MTLBuffer>)tasks_buf_.GetMTLBuffer()
                offset:0
               atIndex:0];
    [encoder setBytes:&num_tasks length:sizeof(num_tasks) atIndex:1];
    [encoder setBuffer:(__bridge id<MTLBuffer>)is_feature_used_buf_.GetMTLBuffer()
                offset:0
               atIndex:2];
    [encoder setBuffer:(__bridge id<MTLBuffer>)histogram_input_buf_.GetMTLBuffer()
                offset:hist_offset
               atIndex:3];
    [encoder setBytes:&total_grad length:sizeof(total_grad) atIndex:4];
    [encoder setBytes:&total_hess length:sizeof(total_hess) atIndex:5];
    [encoder setBytes:&total_count length:sizeof(total_count) atIndex:6];
    [encoder setBytes:&parent_gain length:sizeof(parent_gain) atIndex:7];
    [encoder setBytes:&lambda_l1 length:sizeof(lambda_l1) atIndex:8];
    [encoder setBytes:&lambda_l2 length:sizeof(lambda_l2) atIndex:9];
    [encoder setBytes:&min_gain_to_split length:sizeof(min_gain_to_split) atIndex:10];
    [encoder setBytes:&min_data_in_leaf length:sizeof(min_data_in_leaf) atIndex:11];
    [encoder setBytes:&min_sum_hessian_in_leaf
                length:sizeof(min_sum_hessian_in_leaf)
               atIndex:12];
    [encoder setBuffer:(__bridge id<MTLBuffer>)per_task_result_buf_.GetMTLBuffer()
                offset:0
               atIndex:13];
    [encoder dispatchThreadgroups:MTLSizeMake(num_tasks_, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];
    METAL_CHECK([cmd_buf status] != MTLCommandBufferStatusError,
                [[[cmd_buf error] localizedDescription] UTF8String]);
  }

  MetalSplitResult best{};
  best.gain = -std::numeric_limits<float>::infinity();
  best.feature = -1;
  best.found = 0;

  const MetalSplitResult* results = per_task_result_buf_.data();
  for (int t = 0; t < num_tasks_; ++t) {
    const MetalSplitResult& candidate = results[t];
    if (!candidate.found) {
      continue;
    }
    if (candidate.feature < 0 || candidate.feature >= num_features_) {
      continue;
    }
    if (!node_feature_mask[candidate.feature]) {
      continue;
    }
    if (IsBetterResult(candidate, best)) {
      best = candidate;
    }
  }

  per_leaf_best_buf_.data()[leaf_index] = best;
}

void MetalBestSplitFinder::FindBestSplitsForLeaf(
    const hist_t* smaller_leaf_hist,
    const LeafSplits* smaller_leaf_splits,
    int smaller_leaf_index,
    const std::vector<int8_t>& smaller_node_used_features,
    const hist_t* larger_leaf_hist,
    const LeafSplits* larger_leaf_splits,
    int larger_leaf_index,
    const std::vector<int8_t>* larger_node_used_features) {
  if (smaller_leaf_hist != nullptr && smaller_leaf_splits != nullptr) {
    UploadHistogram(smaller_leaf_hist, 0);
    DispatchSplitKernel(smaller_leaf_splits, smaller_leaf_index, 0,
                        smaller_node_used_features);
  } else {
    ClearLeafBest(smaller_leaf_index);
  }

  if (larger_leaf_index >= 0 && larger_leaf_hist != nullptr &&
      larger_leaf_splits != nullptr && larger_node_used_features != nullptr) {
    UploadHistogram(larger_leaf_hist, 1);
    DispatchSplitKernel(larger_leaf_splits, larger_leaf_index, 1,
                        *larger_node_used_features);
  } else {
    ClearLeafBest(larger_leaf_index);
  }
}

void MetalBestSplitFinder::GetBestSplitForLeaf(
    int leaf_index,
    SplitInfo* best_split) const {
  best_split->Reset();
  if (leaf_index < 0 || leaf_index >= num_leaves_) {
    return;
  }

  const MetalSplitResult& result = per_leaf_best_buf_.data()[leaf_index];
  if (!result.found) {
    return;
  }

  best_split->feature = result.feature;
  best_split->threshold = result.threshold;
  best_split->default_left = (result.default_left != 0);
  best_split->gain = result.gain;
  best_split->left_sum_gradient = result.left_sum_gradient;
  best_split->left_sum_hessian = result.left_sum_hessian;
  best_split->left_count = static_cast<data_size_t>(result.left_count);
  best_split->right_sum_gradient = result.right_sum_gradient;
  best_split->right_sum_hessian = result.right_sum_hessian;
  best_split->right_count = static_cast<data_size_t>(result.right_count);
  best_split->left_output = result.left_value;
  best_split->right_output = result.right_value;
  best_split->num_cat_threshold = 0;
}

void MetalBestSplitFinder::FindBestFromAllSplits(
    int /*cur_num_leaves*/,
    int smaller_leaf_index,
    int larger_leaf_index,
    int* best_leaf_index,
    SplitInfo* best_split) const {
  *best_leaf_index = -1;
  best_split->Reset();

  int winner = -1;
  MetalSplitResult best{};
  best.gain = -std::numeric_limits<float>::infinity();
  best.feature = -1;
  best.found = 0;

  if (smaller_leaf_index >= 0 &&
      IsBetterResult(per_leaf_best_buf_.data()[smaller_leaf_index], best)) {
    best = per_leaf_best_buf_.data()[smaller_leaf_index];
    winner = smaller_leaf_index;
  }
  if (larger_leaf_index >= 0 &&
      IsBetterResult(per_leaf_best_buf_.data()[larger_leaf_index], best)) {
    best = per_leaf_best_buf_.data()[larger_leaf_index];
    winner = larger_leaf_index;
  }

  if (winner < 0) {
    return;
  }

  *best_leaf_index = winner;
  GetBestSplitForLeaf(winner, best_split);
}

void MetalBestSplitFinder::ResetTrainingData(
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets) {
  num_features_ = train_data->num_features();
  num_total_bin_ = feature_hist_offsets.empty()
                       ? 0
                       : static_cast<int>(feature_hist_offsets.back());
  feature_hist_offsets_ = feature_hist_offsets;
  InitFeatureMetaInfo(train_data);
  InitTasks();
  is_feature_used_buf_.Resize(static_cast<size_t>(num_features_));
  per_task_result_buf_.Resize(static_cast<size_t>(num_tasks_));
  per_leaf_best_buf_.Resize(static_cast<size_t>(num_leaves_));
  histogram_input_buf_.Resize(static_cast<size_t>(num_total_bin_) * 2 * 2);
}

void MetalBestSplitFinder::ResetConfig(const Config* config) {
  num_leaves_ = config->num_leaves;
  lambda_l1_ = config->lambda_l1;
  lambda_l2_ = config->lambda_l2;
  min_data_in_leaf_ = config->min_data_in_leaf;
  min_sum_hessian_in_leaf_ = config->min_sum_hessian_in_leaf;
  min_gain_to_split_ = config->min_gain_to_split;
  per_leaf_best_buf_.Resize(static_cast<size_t>(num_leaves_));
}

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
