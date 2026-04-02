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
#include <vector>

namespace LightGBM {

// ---------------------------------------------------------------------------
//  Construction / destruction
// ---------------------------------------------------------------------------

MetalBestSplitFinder::MetalBestSplitFinder(
    const hist_t* hist_data,
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets,
    const Config* config)
    : num_features_(train_data->num_features()),
      num_leaves_(config->num_leaves),
      feature_hist_offsets_(feature_hist_offsets),
      lambda_l1_(config->lambda_l1),
      lambda_l2_(config->lambda_l2),
      min_data_in_leaf_(config->min_data_in_leaf),
      min_sum_hessian_in_leaf_(config->min_sum_hessian_in_leaf),
      min_gain_to_split_(config->min_gain_to_split),
      cat_smooth_(config->cat_smooth),
      cat_l2_(config->cat_l2),
      max_cat_threshold_(config->max_cat_threshold),
      min_data_per_group_(config->min_data_per_group),
      max_cat_to_onehot_(config->max_cat_to_onehot),
      has_categorical_feature_(false),
      max_num_categorical_bin_(0),
      num_tasks_(0),
      hist_data_(hist_data),
      find_best_split_pso_(nullptr),
      sync_best_split_pso_(nullptr),
      find_best_from_all_pso_(nullptr) {
  InitFeatureMetaInfo(train_data);
}

MetalBestSplitFinder::~MetalBestSplitFinder() {}

// ---------------------------------------------------------------------------
//  Feature metadata
// ---------------------------------------------------------------------------

void MetalBestSplitFinder::InitFeatureMetaInfo(const Dataset* train_data) {
  feature_missing_type_.resize(num_features_);
  feature_mfb_offsets_.resize(num_features_);
  feature_default_bins_.resize(num_features_);
  feature_num_bins_.resize(num_features_);
  has_categorical_feature_ = false;
  max_num_categorical_bin_ = 0;
  is_categorical_.resize(num_features_, 0);

  for (int f = 0; f < num_features_; ++f) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(f);
    if (bin_mapper->bin_type() == BinType::CategoricalBin) {
      has_categorical_feature_ = true;
      is_categorical_[f] = 1;
      if (bin_mapper->num_bin() > max_num_categorical_bin_) {
        max_num_categorical_bin_ = bin_mapper->num_bin();
      }
    }
    feature_missing_type_[f] = bin_mapper->missing_type();
    feature_mfb_offsets_[f] =
        static_cast<uint8_t>(bin_mapper->GetMostFreqBin() == 0);
    feature_default_bins_[f] = bin_mapper->GetDefaultBin();
    feature_num_bins_[f] = static_cast<uint32_t>(bin_mapper->num_bin());
  }
}

// ---------------------------------------------------------------------------
//  Task construction (mirrors CUDA InitCUDAFeatureMetaInfo)
// ---------------------------------------------------------------------------

void MetalBestSplitFinder::InitTasks() {
  num_tasks_ = 0;
  for (int f = 0; f < num_features_; ++f) {
    const uint32_t num_bin = feature_num_bins_[f];
    const MissingType mt = feature_missing_type_[f];
    if (num_bin > 2 && mt != MissingType::None && !is_categorical_[f]) {
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
    if (num_bin > 2 && mt != MissingType::None && !is_categorical_[f]) {
      // Forward pass
      MetalSplitFindTask& t0 = split_find_tasks_[cur];
      t0.inner_feature_index = f;
      t0.reverse = (mt == MissingType::Zero) ? 0 : 0;
      t0.skip_default_bin = (mt == MissingType::Zero) ? 1 : 0;
      t0.na_as_missing = (mt == MissingType::NaN) ? 1 : 0;
      t0.assume_out_default_left = 0;
      t0.is_categorical = 0;
      t0.is_one_hot = 0;
      t0.hist_offset = feature_hist_offsets_[f];
      t0.mfb_offset = feature_mfb_offsets_[f];
      t0.default_bin = feature_default_bins_[f];
      t0.num_bin = num_bin;
      ++cur;

      // Reverse pass
      MetalSplitFindTask& t1 = split_find_tasks_[cur];
      t1.inner_feature_index = f;
      t1.reverse = 1;
      t1.skip_default_bin = (mt == MissingType::Zero) ? 1 : 0;
      t1.na_as_missing = (mt == MissingType::NaN) ? 1 : 0;
      t1.assume_out_default_left = 1;
      t1.is_categorical = 0;
      t1.is_one_hot = 0;
      t1.hist_offset = feature_hist_offsets_[f];
      t1.mfb_offset = feature_mfb_offsets_[f];
      t1.default_bin = feature_default_bins_[f];
      t1.num_bin = num_bin;
      ++cur;
    } else {
      MetalSplitFindTask& t = split_find_tasks_[cur];
      t.inner_feature_index = f;
      if (is_categorical_[f]) {
        t.reverse = 0;
        t.is_categorical = 1;
        t.is_one_hot =
            (static_cast<int>(num_bin) <= max_cat_to_onehot_) ? 1 : 0;
      } else {
        t.reverse = 1;
        t.is_categorical = 0;
        t.is_one_hot = 0;
      }
      t.skip_default_bin = 0;
      t.na_as_missing = 0;
      if (mt != MissingType::NaN && !is_categorical_[f]) {
        t.assume_out_default_left = 1;
      } else {
        t.assume_out_default_left = 0;
      }
      t.hist_offset = feature_hist_offsets_[f];
      t.mfb_offset = feature_mfb_offsets_[f];
      t.default_bin = feature_default_bins_[f];
      t.num_bin = num_bin;
      ++cur;
    }
  }

  // Upload tasks to Metal buffer.
  tasks_buf_.Resize(static_cast<size_t>(num_tasks_));
  std::memcpy(tasks_buf_.data(), split_find_tasks_.data(),
              num_tasks_ * sizeof(MetalSplitFindTask));
}

// ---------------------------------------------------------------------------
//  Init
// ---------------------------------------------------------------------------

void MetalBestSplitFinder::Init() {
  InitTasks();

  is_feature_used_buf_.Resize(static_cast<size_t>(num_features_));

  // Per-task results for both smaller and larger leaves.
  const size_t result_size = static_cast<size_t>(num_tasks_) * 2;
  per_task_result_buf_.Resize(result_size);

  // Per-leaf best splits.
  per_leaf_best_buf_.Resize(static_cast<size_t>(num_leaves_));

  find_best_split_pso_ = MetalDevice::GetPipeline("find_best_split");
  sync_best_split_pso_ = MetalDevice::GetPipeline("sync_best_split");
  find_best_from_all_pso_ = MetalDevice::GetPipeline("find_best_from_all");
}

// ---------------------------------------------------------------------------
//  BeforeTrain
// ---------------------------------------------------------------------------

void MetalBestSplitFinder::BeforeTrain(
    const std::vector<int8_t>& is_feature_used_bytree) {
  std::memcpy(is_feature_used_buf_.data(), is_feature_used_bytree.data(),
              is_feature_used_bytree.size() * sizeof(int8_t));

  // Zero per-leaf best.
  MetalSplitResult* leaf_best = per_leaf_best_buf_.data();
  for (int i = 0; i < num_leaves_; ++i) {
    leaf_best[i].gain = kMinScore;
    leaf_best[i].found = 0;
  }
}

// ---------------------------------------------------------------------------
//  CPU-side split evaluation helpers
// ---------------------------------------------------------------------------

static double ThresholdL1(double s, double l1) {
  const double reg_s = std::fmax(0.0, std::fabs(s) - l1);
  return (s >= 0.0) ? reg_s : -reg_s;
}

static double CalcLeafOutput(double sum_grad, double sum_hess,
                             double l1, double l2) {
  if (l1 > 0.0) {
    return -ThresholdL1(sum_grad, l1) / (sum_hess + l2);
  }
  return -sum_grad / (sum_hess + l2);
}

static double CalcLeafGain(double sum_grad, double sum_hess,
                           double l1, double l2) {
  if (l1 > 0.0) {
    const double sg = ThresholdL1(sum_grad, l1);
    return (sg * sg) / (sum_hess + l2);
  }
  return (sum_grad * sum_grad) / (sum_hess + l2);
}

static double CalcSplitGain(double left_grad, double left_hess,
                            double right_grad, double right_hess,
                            double l1, double l2) {
  return CalcLeafGain(left_grad, left_hess, l1, l2) +
         CalcLeafGain(right_grad, right_hess, l1, l2);
}

// ---------------------------------------------------------------------------
//  CPU-side per-feature best split scan
// ---------------------------------------------------------------------------

static void FindBestSplitForTask(
    const MetalSplitFindTask& task,
    const hist_t* leaf_hist,
    double total_grad, double total_hess,
    data_size_t total_cnt,
    double l1, double l2,
    data_size_t min_data_in_leaf,
    double min_sum_hessian_in_leaf,
    double min_gain_to_split,
    double parent_gain,
    MetalSplitResult* result) {
  result->gain = kMinScore;
  result->found = 0;

  if (task.is_categorical) {
    // Categorical split evaluation: one-hot style (each bin vs rest).
    const uint32_t offset = task.hist_offset;
    const uint32_t num_bin = task.num_bin;
    const uint8_t mfb_offset = task.mfb_offset;

    double best_gain = kMinScore;
    uint32_t best_threshold = 0;

    for (uint32_t b = 0; b < num_bin; ++b) {
      if (b == 0 && mfb_offset == 0) continue;
      const uint32_t idx = (offset + b - mfb_offset) * 2;
      const double bin_grad = leaf_hist[idx];
      const double bin_hess = leaf_hist[idx + 1];
      const data_size_t bin_cnt = (bin_hess > 0.0) ?
          static_cast<data_size_t>(bin_hess + 0.5) : 0;

      const double other_grad = total_grad - bin_grad;
      const double other_hess = total_hess - bin_hess;
      const data_size_t other_cnt = total_cnt - bin_cnt;

      if (bin_cnt < min_data_in_leaf || bin_hess < min_sum_hessian_in_leaf)
        continue;
      if (other_cnt < min_data_in_leaf ||
          other_hess < min_sum_hessian_in_leaf)
        continue;

      const double gain =
          CalcSplitGain(bin_grad, bin_hess, other_grad, other_hess, l1, l2) -
          parent_gain;
      if (gain > best_gain) {
        best_gain = gain;
        best_threshold = b;
      }
    }

    if (best_gain > min_gain_to_split) {
      const uint32_t idx =
          (offset + best_threshold - mfb_offset) * 2;
      const double left_grad = leaf_hist[idx];
      const double left_hess = leaf_hist[idx + 1];
      result->gain = best_gain;
      result->feature = task.inner_feature_index;
      result->threshold = best_threshold;
      result->default_left = 0;
      result->left_sum_gradient = left_grad;
      result->left_sum_hessian = left_hess;
      result->left_count =
          (left_hess > 0.0) ? static_cast<data_size_t>(left_hess + 0.5) : 0;
      result->right_sum_gradient = total_grad - left_grad;
      result->right_sum_hessian = total_hess - left_hess;
      result->right_count = total_cnt - result->left_count;
      result->left_value =
          CalcLeafOutput(left_grad, left_hess, l1, l2);
      result->right_value = CalcLeafOutput(
          result->right_sum_gradient, result->right_sum_hessian, l1, l2);
      result->found = 1;
    }
    return;
  }

  // Numerical feature: scan bins in one direction, accumulating partial sums.
  const uint32_t offset = task.hist_offset;
  const uint32_t num_bin = task.num_bin;
  const uint8_t mfb_offset = task.mfb_offset;
  const uint32_t default_bin = task.default_bin;

  double best_gain = kMinScore;
  uint32_t best_bin = 0;
  double best_left_grad = 0.0;
  double best_left_hess = 0.0;
  data_size_t best_left_cnt = 0;

  double prefix_grad = 0.0;
  double prefix_hess = 0.0;
  data_size_t prefix_cnt = 0;

  const uint32_t start_bin = mfb_offset;
  const uint32_t end_bin = num_bin - 1 + mfb_offset;

  if (!task.reverse) {
    // Forward scan (left to right).
    for (uint32_t b = start_bin; b < end_bin; ++b) {
      if (task.skip_default_bin && (b + (1 - mfb_offset)) == default_bin)
        continue;

      const uint32_t idx = (offset + b) * 2;
      const double bin_grad = leaf_hist[idx];
      const double bin_hess = leaf_hist[idx + 1];
      const data_size_t bin_cnt =
          (bin_hess > 0.0) ? static_cast<data_size_t>(bin_hess + 0.5) : 0;

      prefix_grad += bin_grad;
      prefix_hess += bin_hess;
      prefix_cnt += bin_cnt;

      const double right_grad = total_grad - prefix_grad;
      const double right_hess = total_hess - prefix_hess;
      const data_size_t right_cnt = total_cnt - prefix_cnt;

      if (prefix_cnt < min_data_in_leaf ||
          prefix_hess < min_sum_hessian_in_leaf)
        continue;
      if (right_cnt < min_data_in_leaf ||
          right_hess < min_sum_hessian_in_leaf)
        continue;

      const double gain = CalcSplitGain(prefix_grad, prefix_hess,
                                        right_grad, right_hess, l1, l2) -
                          parent_gain;
      if (gain > best_gain) {
        best_gain = gain;
        best_bin = b - mfb_offset;
        best_left_grad = prefix_grad;
        best_left_hess = prefix_hess;
        best_left_cnt = prefix_cnt;
      }
    }
  } else {
    // Reverse scan (right to left).
    for (int32_t b = static_cast<int32_t>(end_bin) - 1;
         b >= static_cast<int32_t>(start_bin); --b) {
      if (task.skip_default_bin &&
          (static_cast<uint32_t>(b) + (1 - mfb_offset)) == default_bin)
        continue;

      const uint32_t idx = (offset + static_cast<uint32_t>(b)) * 2;
      const double bin_grad = leaf_hist[idx];
      const double bin_hess = leaf_hist[idx + 1];
      const data_size_t bin_cnt =
          (bin_hess > 0.0) ? static_cast<data_size_t>(bin_hess + 0.5) : 0;

      prefix_grad += bin_grad;
      prefix_hess += bin_hess;
      prefix_cnt += bin_cnt;

      const double right_grad = total_grad - prefix_grad;
      const double right_hess = total_hess - prefix_hess;
      const data_size_t right_cnt = total_cnt - prefix_cnt;

      if (prefix_cnt < min_data_in_leaf ||
          prefix_hess < min_sum_hessian_in_leaf)
        continue;
      if (right_cnt < min_data_in_leaf ||
          right_hess < min_sum_hessian_in_leaf)
        continue;

      const double gain = CalcSplitGain(right_grad, right_hess,
                                        prefix_grad, prefix_hess, l1, l2) -
                          parent_gain;
      if (gain > best_gain) {
        best_gain = gain;
        best_bin = static_cast<uint32_t>(b) - mfb_offset;
        best_left_grad = right_grad;
        best_left_hess = right_hess;
        best_left_cnt = right_cnt;
      }
    }
  }

  if (best_gain > min_gain_to_split) {
    result->gain = best_gain;
    result->feature = task.inner_feature_index;
    result->threshold = best_bin;
    result->default_left = task.assume_out_default_left ? 1 : 0;
    result->left_sum_gradient = best_left_grad;
    result->left_sum_hessian = best_left_hess;
    result->left_count = best_left_cnt;
    result->right_sum_gradient = total_grad - best_left_grad;
    result->right_sum_hessian = total_hess - best_left_hess;
    result->right_count = total_cnt - best_left_cnt;
    result->left_value =
        CalcLeafOutput(best_left_grad, best_left_hess, l1, l2);
    result->right_value = CalcLeafOutput(
        result->right_sum_gradient, result->right_sum_hessian, l1, l2);
    result->found = 1;
  }
}

// ---------------------------------------------------------------------------
//  FindBestSplitsForLeaf
// ---------------------------------------------------------------------------

void MetalBestSplitFinder::FindBestSplitsForLeaf(
    const MetalLeafSplitsStruct* smaller_leaf,
    const MetalLeafSplitsStruct* larger_leaf,
    int smaller_leaf_index,
    int larger_leaf_index,
    data_size_t num_data_in_smaller_leaf,
    data_size_t num_data_in_larger_leaf,
    double sum_hessians_in_smaller_leaf,
    double sum_hessians_in_larger_leaf) {
  const bool is_smaller_valid =
      (num_data_in_smaller_leaf > min_data_in_leaf_ &&
       sum_hessians_in_smaller_leaf > min_sum_hessian_in_leaf_);
  const bool is_larger_valid =
      (num_data_in_larger_leaf > min_data_in_leaf_ &&
       sum_hessians_in_larger_leaf > min_sum_hessian_in_leaf_ &&
       larger_leaf_index >= 0);

  const int8_t* is_used = is_feature_used_buf_.data();

  // Evaluate splits for the smaller leaf.
  if (is_smaller_valid) {
    const hist_t* leaf_hist =
        hist_data_ + smaller_leaf->hist_offset;
    const double total_grad = smaller_leaf->sum_of_gradients;
    const double total_hess = smaller_leaf->sum_of_hessians;
    const data_size_t total_cnt = smaller_leaf->num_data_in_leaf;
    const double parent_gain = smaller_leaf->gain;

    MetalSplitResult* results = per_task_result_buf_.data();

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_tasks_; ++t) {
      const MetalSplitFindTask& task = split_find_tasks_[t];
      results[t].gain = kMinScore;
      results[t].found = 0;
      if (!is_used[task.inner_feature_index]) continue;
      FindBestSplitForTask(task, leaf_hist, total_grad, total_hess,
                           total_cnt, lambda_l1_, lambda_l2_,
                           min_data_in_leaf_, min_sum_hessian_in_leaf_,
                           min_gain_to_split_, parent_gain, &results[t]);
    }

    // Reduce across tasks for the smaller leaf.
    MetalSplitResult& best = per_leaf_best_buf_.data()[smaller_leaf_index];
    best.gain = kMinScore;
    best.found = 0;
    for (int t = 0; t < num_tasks_; ++t) {
      if (results[t].found && results[t].gain > best.gain) {
        best = results[t];
      }
    }
  }

  // Evaluate splits for the larger leaf.
  if (is_larger_valid) {
    const hist_t* leaf_hist =
        hist_data_ + larger_leaf->hist_offset;
    const double total_grad = larger_leaf->sum_of_gradients;
    const double total_hess = larger_leaf->sum_of_hessians;
    const data_size_t total_cnt = larger_leaf->num_data_in_leaf;
    const double parent_gain = larger_leaf->gain;

    MetalSplitResult* results =
        per_task_result_buf_.data() + num_tasks_;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_tasks_; ++t) {
      const MetalSplitFindTask& task = split_find_tasks_[t];
      results[t].gain = kMinScore;
      results[t].found = 0;
      if (!is_used[task.inner_feature_index]) continue;
      FindBestSplitForTask(task, leaf_hist, total_grad, total_hess,
                           total_cnt, lambda_l1_, lambda_l2_,
                           min_data_in_leaf_, min_sum_hessian_in_leaf_,
                           min_gain_to_split_, parent_gain, &results[t]);
    }

    MetalSplitResult& best = per_leaf_best_buf_.data()[larger_leaf_index];
    best.gain = kMinScore;
    best.found = 0;
    for (int t = 0; t < num_tasks_; ++t) {
      if (results[t].found && results[t].gain > best.gain) {
        best = results[t];
      }
    }
  }
}

// ---------------------------------------------------------------------------
//  FindBestFromAllSplits — find the single best split across all leaves
// ---------------------------------------------------------------------------

void MetalBestSplitFinder::FindBestFromAllSplits(
    int cur_num_leaves,
    int smaller_leaf_index,
    int larger_leaf_index,
    int* best_leaf_index,
    SplitInfo* best_split) {
  *best_leaf_index = -1;
  best_split->gain = kMinScore;

  const MetalSplitResult* leaf_best = per_leaf_best_buf_.data();

  double best_gain = kMinScore;
  int best_leaf = -1;

  // Check the two candidate leaves (smaller and larger).
  if (leaf_best[smaller_leaf_index].found &&
      leaf_best[smaller_leaf_index].gain > best_gain) {
    best_gain = leaf_best[smaller_leaf_index].gain;
    best_leaf = smaller_leaf_index;
  }
  if (larger_leaf_index >= 0 &&
      leaf_best[larger_leaf_index].found &&
      leaf_best[larger_leaf_index].gain > best_gain) {
    best_gain = leaf_best[larger_leaf_index].gain;
    best_leaf = larger_leaf_index;
  }

  if (best_leaf < 0) {
    return;
  }

  const MetalSplitResult& r = leaf_best[best_leaf];
  *best_leaf_index = best_leaf;
  best_split->feature = r.feature;
  best_split->threshold = r.threshold;
  best_split->default_left = (r.default_left != 0);
  best_split->gain = r.gain;
  best_split->left_sum_gradient = r.left_sum_gradient;
  best_split->left_sum_hessian = r.left_sum_hessian;
  best_split->left_count = r.left_count;
  best_split->right_sum_gradient = r.right_sum_gradient;
  best_split->right_sum_hessian = r.right_sum_hessian;
  best_split->right_count = r.right_count;
  best_split->left_output = r.left_value;
  best_split->right_output = r.right_value;
  best_split->num_cat_threshold = 0;
}

// ---------------------------------------------------------------------------
//  ResetTrainingData
// ---------------------------------------------------------------------------

void MetalBestSplitFinder::ResetTrainingData(
    const hist_t* hist_data,
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets) {
  hist_data_ = hist_data;
  num_features_ = train_data->num_features();
  feature_hist_offsets_ = feature_hist_offsets;
  InitFeatureMetaInfo(train_data);
  InitTasks();

  is_feature_used_buf_.Resize(static_cast<size_t>(num_features_));

  const size_t result_size = static_cast<size_t>(num_tasks_) * 2;
  per_task_result_buf_.Resize(result_size);
  per_leaf_best_buf_.Resize(static_cast<size_t>(num_leaves_));
}

// ---------------------------------------------------------------------------
//  ResetConfig
// ---------------------------------------------------------------------------

void MetalBestSplitFinder::ResetConfig(const Config* config,
                                       const hist_t* hist_data) {
  num_leaves_ = config->num_leaves;
  lambda_l1_ = config->lambda_l1;
  lambda_l2_ = config->lambda_l2;
  min_data_in_leaf_ = config->min_data_in_leaf;
  min_sum_hessian_in_leaf_ = config->min_sum_hessian_in_leaf;
  min_gain_to_split_ = config->min_gain_to_split;
  cat_smooth_ = config->cat_smooth;
  cat_l2_ = config->cat_l2;
  max_cat_threshold_ = config->max_cat_threshold;
  min_data_per_group_ = config->min_data_per_group;
  max_cat_to_onehot_ = config->max_cat_to_onehot;
  hist_data_ = hist_data;
  per_leaf_best_buf_.Resize(static_cast<size_t>(num_leaves_));
}

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
