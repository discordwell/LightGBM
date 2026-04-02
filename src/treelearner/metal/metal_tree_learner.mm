/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef LGBM_USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_tree_learner.hpp"

#include <LightGBM/feature_group.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/common.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace LightGBM {

// ---------------------------------------------------------------------------
//  Gain helper (same as CUDALeafSplits::GetLeafGainGivenOutput on CPU)
// ---------------------------------------------------------------------------

static double CalcLeafGainGivenOutput(double sum_gradients,
                                      double sum_hessians,
                                      double l1, double l2,
                                      double output) {
  const double sg = std::fmax(0.0, std::fabs(sum_gradients) - l1);
  const double signed_sg = (sum_gradients >= 0.0) ? sg : -sg;
  return -(2.0 * signed_sg * output + (sum_hessians + l2) * output * output);
}

// ---------------------------------------------------------------------------
//  Construction / destruction
// ---------------------------------------------------------------------------

MetalSingleGPUTreeLearner::MetalSingleGPUTreeLearner(const Config* config)
    : SerialTreeLearner(config),
      smaller_leaf_index_(0),
      larger_leaf_index_(-1),
      best_leaf_index_(-1),
      has_categorical_feature_(false),
      num_threads_(1) {}

MetalSingleGPUTreeLearner::~MetalSingleGPUTreeLearner() {}

// ---------------------------------------------------------------------------
//  Init
// ---------------------------------------------------------------------------

void MetalSingleGPUTreeLearner::Init(const Dataset* train_data,
                                     bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();

  // Leaf splits (root sums computed on CPU via OpenMP reduction).
  smaller_leaf_splits_.reset(new MetalLeafSplits(num_data_));
  larger_leaf_splits_.reset(new MetalLeafSplits(num_data_));

  // Histogram constructor.
  const auto& feature_hist_offsets = share_state_->feature_hist_offsets();
  histogram_constructor_.reset(new MetalHistogramConstructor(
      train_data_, config_->num_leaves, num_threads_, feature_hist_offsets,
      config_->min_data_in_leaf, config_->min_sum_hessian_in_leaf));
  histogram_constructor_->Init(train_data_, share_state_.get());

  // Best split finder.
  best_split_finder_.reset(new MetalBestSplitFinder(
      histogram_constructor_->hist_data(), train_data_,
      feature_hist_offsets, config_));
  best_split_finder_->Init();

  // Per-leaf tracking vectors.
  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_gradients_.resize(config_->num_leaves, 0.0);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0);

  // Detect categorical features.
  has_categorical_feature_ = false;
  for (int i = 0; i < train_data_->num_features(); ++i) {
    if (train_data_->FeatureBinMapper(i)->bin_type() ==
        BinType::CategoricalBin) {
      has_categorical_feature_ = true;
      break;
    }
  }
}

// ---------------------------------------------------------------------------
//  BeforeTrain
// ---------------------------------------------------------------------------

void MetalSingleGPUTreeLearner::BeforeTrain() {
  fprintf(stderr, "[Metal] BeforeTrain: start\n");

  data_partition_->Init();
  fprintf(stderr, "[Metal] BeforeTrain: data_partition init done\n");

  const data_size_t root_num_data = data_partition_->leaf_count(0);

  // Initialise root leaf splits on CPU.
  data_size_t root_cnt_check = 0;
  const data_size_t* root_indices =
      data_partition_->GetIndexOnLeaf(0, &root_cnt_check);
  smaller_leaf_splits_->Init(
      gradients_, hessians_, root_indices, root_num_data,
      histogram_constructor_->hist_data(), config_->lambda_l1,
      config_->lambda_l2);

  // Record root statistics.
  const MetalLeafSplitsStruct* root_struct =
      smaller_leaf_splits_->GetStruct();
  leaf_sum_gradients_[0] = root_struct->sum_of_gradients;
  leaf_sum_hessians_[0] = root_struct->sum_of_hessians;
  leaf_num_data_[0] = root_num_data;
  leaf_data_start_[0] = 0;

  // Mark the larger leaf as invalid.
  larger_leaf_splits_->InitValues();

  // Prepare histogram and split finder for the iteration.
  fprintf(stderr, "[Metal] BeforeTrain: root leaf init done, grad_sum=%.4f hess_sum=%.4f num_data=%d\n",
          root_struct->sum_of_gradients, root_struct->sum_of_hessians, root_num_data);
  histogram_constructor_->BeforeTrain(gradients_, hessians_);
  fprintf(stderr, "[Metal] BeforeTrain: histogram constructor ready\n");

  col_sampler_.ResetByTree();
  best_split_finder_->BeforeTrain(col_sampler_.is_feature_used_bytree());

  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;
}

// ---------------------------------------------------------------------------
//  Train
// ---------------------------------------------------------------------------

Tree* MetalSingleGPUTreeLearner::Train(const score_t* gradients,
                                       const score_t* hessians,
                                       bool /*is_first_tree*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  BeforeTrain();

  const bool track_branch_features =
      !(config_->interaction_constraints_vector.empty());
  auto tree = std::unique_ptr<Tree>(
      new Tree(config_->num_leaves, track_branch_features,
               config_->linear_tree));

  // Set the root leaf output.
  const MetalLeafSplitsStruct* root = smaller_leaf_splits_->GetStruct();
  tree->SetLeafOutput(0, root->leaf_value);

  fprintf(stderr, "[Metal] Train: entering split loop, num_leaves=%d\n", config_->num_leaves);
  // Main split loop.
  for (int i = 0; i < config_->num_leaves - 1; ++i) {
    fprintf(stderr, "[Metal] Train: split %d, smaller=%d larger=%d\n", i, smaller_leaf_index_, larger_leaf_index_);
    // --- Histogram construction ---
    const data_size_t num_data_smaller = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_larger =
        (larger_leaf_index_ < 0) ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hess_smaller = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hess_larger =
        (larger_leaf_index_ < 0) ? 0.0
                                 : leaf_sum_hessians_[larger_leaf_index_];

    // Build histogram for the smaller leaf on the GPU.
    histogram_constructor_->ConstructHistogramForLeaf(
        smaller_leaf_splits_->GetStruct(),
        larger_leaf_splits_->GetStruct(),
        num_data_smaller, num_data_larger,
        sum_hess_smaller, sum_hess_larger);

    // Histogram subtraction: larger = parent - smaller.
    histogram_constructor_->SubtractHistogramForLeaf(
        smaller_leaf_splits_->GetStruct(),
        larger_leaf_splits_->GetStruct());

    // --- Find best splits ---
    best_split_finder_->FindBestSplitsForLeaf(
        smaller_leaf_splits_->GetStruct(),
        larger_leaf_splits_->GetStruct(),
        smaller_leaf_index_, larger_leaf_index_,
        num_data_smaller, num_data_larger,
        sum_hess_smaller, sum_hess_larger);

    SplitInfo best_split;
    best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(), smaller_leaf_index_, larger_leaf_index_,
        &best_leaf_index_, &best_split);

    if (best_leaf_index_ == -1) {
      Log::Warning("No further splits with positive gain, "
                   "training stopped with %d leaves.", (i + 1));
      break;
    }

    // --- Apply the split ---
    const int inner_feature_index = best_split.feature;
    const int real_feature_index =
        train_data_->RealFeatureIndex(inner_feature_index);
    const bool is_numerical =
        train_data_->FeatureBinMapper(inner_feature_index)->bin_type() ==
        BinType::NumericalBin;

    const int next_leaf_id = tree->NextLeafId();
    int right_leaf_index = 0;
    if (is_numerical) {
      const double threshold_double = train_data_->RealThreshold(
          inner_feature_index, best_split.threshold);
      data_partition_->Split(
          best_leaf_index_, train_data_, inner_feature_index,
          &best_split.threshold, 1, best_split.default_left,
          next_leaf_id);
      best_split.left_count =
          data_partition_->leaf_count(best_leaf_index_);
      best_split.right_count =
          data_partition_->leaf_count(next_leaf_id);

      right_leaf_index = tree->Split(
          best_leaf_index_, inner_feature_index, real_feature_index,
          best_split.threshold, threshold_double,
          static_cast<double>(best_split.left_output),
          static_cast<double>(best_split.right_output),
          static_cast<data_size_t>(best_split.left_count),
          static_cast<data_size_t>(best_split.right_count),
          static_cast<double>(best_split.left_sum_hessian),
          static_cast<double>(best_split.right_sum_hessian),
          static_cast<float>(best_split.gain + config_->min_gain_to_split),
          train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
          best_split.default_left);
    } else {
      // Categorical split.
      std::vector<uint32_t> cat_bitset_inner =
          Common::ConstructBitset(best_split.cat_threshold.data(),
                                  best_split.num_cat_threshold);
      std::vector<int> threshold_int(best_split.num_cat_threshold);
      for (int j = 0; j < best_split.num_cat_threshold; ++j) {
        threshold_int[j] = static_cast<int>(train_data_->RealThreshold(
            inner_feature_index, best_split.cat_threshold[j]));
      }
      std::vector<uint32_t> cat_bitset =
          Common::ConstructBitset(threshold_int.data(),
                                  best_split.num_cat_threshold);

      data_partition_->Split(
          best_leaf_index_, train_data_, inner_feature_index,
          cat_bitset_inner.data(),
          static_cast<int>(cat_bitset_inner.size()),
          best_split.default_left, next_leaf_id);
      best_split.left_count =
          data_partition_->leaf_count(best_leaf_index_);
      best_split.right_count =
          data_partition_->leaf_count(next_leaf_id);

      right_leaf_index = tree->SplitCategorical(
          best_leaf_index_, inner_feature_index, real_feature_index,
          cat_bitset_inner.data(),
          static_cast<int>(cat_bitset_inner.size()),
          cat_bitset.data(), static_cast<int>(cat_bitset.size()),
          static_cast<double>(best_split.left_output),
          static_cast<double>(best_split.right_output),
          static_cast<data_size_t>(best_split.left_count),
          static_cast<data_size_t>(best_split.right_count),
          static_cast<double>(best_split.left_sum_hessian),
          static_cast<double>(best_split.right_sum_hessian),
          static_cast<float>(best_split.gain + config_->min_gain_to_split),
          train_data_->FeatureBinMapper(inner_feature_index)->missing_type());
    }

    // --- Update per-leaf tracking ---
    leaf_best_split_feature_[best_leaf_index_] = best_split.feature;
    leaf_best_split_threshold_[best_leaf_index_] = best_split.threshold;
    leaf_best_split_default_left_[best_leaf_index_] =
        best_split.default_left ? 1 : 0;

    // Update counts from the CPU data partition.
    leaf_num_data_[best_leaf_index_] =
        data_partition_->leaf_count(best_leaf_index_);
    leaf_num_data_[right_leaf_index] =
        data_partition_->leaf_count(right_leaf_index);
    leaf_data_start_[best_leaf_index_] =
        data_partition_->leaf_begin(best_leaf_index_);
    leaf_data_start_[right_leaf_index] =
        data_partition_->leaf_begin(right_leaf_index);
    leaf_sum_gradients_[best_leaf_index_] = best_split.left_sum_gradient;
    leaf_sum_gradients_[right_leaf_index] = best_split.right_sum_gradient;
    leaf_sum_hessians_[best_leaf_index_] = best_split.left_sum_hessian;
    leaf_sum_hessians_[right_leaf_index] = best_split.right_sum_hessian;

    // Determine smaller/larger leaf for the next iteration.
    if (leaf_num_data_[best_leaf_index_] <
        leaf_num_data_[right_leaf_index]) {
      smaller_leaf_index_ = best_leaf_index_;
      larger_leaf_index_ = right_leaf_index;
    } else {
      smaller_leaf_index_ = right_leaf_index;
      larger_leaf_index_ = best_leaf_index_;
    }

    // Update leaf split structs for next iteration.
    {
      MetalLeafSplitsStruct* ss = smaller_leaf_splits_->GetStruct();
      ss->leaf_index = smaller_leaf_index_;
      ss->sum_of_gradients = leaf_sum_gradients_[smaller_leaf_index_];
      ss->sum_of_hessians = leaf_sum_hessians_[smaller_leaf_index_];
      ss->num_data_in_leaf = leaf_num_data_[smaller_leaf_index_];
      ss->data_indices_offset = leaf_data_start_[smaller_leaf_index_];
      ss->hist_offset =
          static_cast<int64_t>(smaller_leaf_index_) *
          static_cast<int64_t>(histogram_constructor_->num_total_bin()) * 2;
      ss->gain = CalcLeafGainGivenOutput(
          ss->sum_of_gradients, ss->sum_of_hessians,
          config_->lambda_l1, config_->lambda_l2,
          tree->LeafOutput(smaller_leaf_index_));
      ss->leaf_value = tree->LeafOutput(smaller_leaf_index_);
    }

    {
      MetalLeafSplitsStruct* ls = larger_leaf_splits_->GetStruct();
      ls->leaf_index = larger_leaf_index_;
      ls->sum_of_gradients = leaf_sum_gradients_[larger_leaf_index_];
      ls->sum_of_hessians = leaf_sum_hessians_[larger_leaf_index_];
      ls->num_data_in_leaf = leaf_num_data_[larger_leaf_index_];
      ls->data_indices_offset = leaf_data_start_[larger_leaf_index_];
      ls->hist_offset =
          static_cast<int64_t>(larger_leaf_index_) *
          static_cast<int64_t>(histogram_constructor_->num_total_bin()) * 2;
      ls->gain = CalcLeafGainGivenOutput(
          ls->sum_of_gradients, ls->sum_of_hessians,
          config_->lambda_l1, config_->lambda_l2,
          tree->LeafOutput(larger_leaf_index_));
      ls->leaf_value = tree->LeafOutput(larger_leaf_index_);
    }
  }

  return tree.release();
}

// ---------------------------------------------------------------------------
//  ResetTrainingData
// ---------------------------------------------------------------------------

void MetalSingleGPUTreeLearner::ResetTrainingData(
    const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);

  histogram_constructor_->ResetTrainingData(train_data, share_state_.get());
  best_split_finder_->ResetTrainingData(
      histogram_constructor_->hist_data(), train_data,
      share_state_->feature_hist_offsets());
}

// ---------------------------------------------------------------------------
//  SetBaggingData
// ---------------------------------------------------------------------------

void MetalSingleGPUTreeLearner::SetBaggingData(
    const Dataset* subset,
    const data_size_t* used_indices,
    data_size_t num_data) {
  if (subset == nullptr) {
    data_partition_->SetUsedDataIndices(used_indices, num_data);
    share_state_->SetUseSubrow(false);
  } else {
    ResetTrainingData(subset, share_state_->is_constant_hessian);
    share_state_->SetUseSubrow(true);
    share_state_->SetSubrowCopied(false);
    share_state_->bagging_use_indices = used_indices;
    share_state_->bagging_indices_cnt = num_data;
  }
}

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
