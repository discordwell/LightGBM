/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_BEST_SPLIT_FINDER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_BEST_SPLIT_FINDER_HPP_

#ifdef LGBM_USE_METAL

#include "metal_utils.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>

#include <vector>

#include "../leaf_splits.hpp"
#include "../split_info.hpp"

namespace LightGBM {

struct MetalSplitFindTask {
  int32_t inner_feature_index;
  int32_t reverse;
  int32_t skip_default_bin;
  int32_t na_as_missing;
  int32_t assume_out_default_left;
  uint32_t hist_offset;
  uint32_t mfb_offset;
  uint32_t num_bin;
  uint32_t default_bin;
};

struct MetalSplitResult {
  float gain;
  int32_t feature;
  uint32_t threshold;
  int32_t default_left;
  float left_sum_gradient;
  float left_sum_hessian;
  int32_t left_count;
  float right_sum_gradient;
  float right_sum_hessian;
  int32_t right_count;
  float left_value;
  float right_value;
  int32_t found;
};

class MetalBestSplitFinder {
 public:
  MetalBestSplitFinder(const Dataset* train_data,
                       const std::vector<uint32_t>& feature_hist_offsets,
                       const Config* config);

  ~MetalBestSplitFinder();

  void Init();

  void BeforeTrain(const std::vector<int8_t>& is_feature_used_bytree);

  void FindBestSplitsForLeaf(
      const hist_t* smaller_leaf_hist,
      const LeafSplits* smaller_leaf_splits,
      int smaller_leaf_index,
      const std::vector<int8_t>& smaller_node_used_features,
      const hist_t* larger_leaf_hist,
      const LeafSplits* larger_leaf_splits,
      int larger_leaf_index,
      const std::vector<int8_t>* larger_node_used_features);

  void GetBestSplitForLeaf(int leaf_index, SplitInfo* best_split) const;

  void FindBestFromAllSplits(
      int cur_num_leaves,
      int smaller_leaf_index,
      int larger_leaf_index,
      int* best_leaf_index,
      SplitInfo* best_split) const;

  void ResetTrainingData(const Dataset* train_data,
                         const std::vector<uint32_t>& feature_hist_offsets);

  void ResetConfig(const Config* config);

 private:
  void InitFeatureMetaInfo(const Dataset* train_data);
  void InitTasks();
  void UploadHistogram(const hist_t* src_hist, size_t slot);
  void DispatchSplitKernel(const LeafSplits* leaf_splits,
                           int leaf_index,
                           size_t hist_slot,
                           const std::vector<int8_t>& node_feature_mask);
  void ClearLeafBest(int leaf_index);

  int num_features_;
  int num_leaves_;
  int num_total_bin_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint32_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_default_bins_;
  std::vector<uint32_t> feature_num_bins_;
  std::vector<MissingType> feature_missing_type_;

  double lambda_l1_;
  double lambda_l2_;
  data_size_t min_data_in_leaf_;
  double min_sum_hessian_in_leaf_;
  double min_gain_to_split_;

  int num_tasks_;
  std::vector<MetalSplitFindTask> split_find_tasks_;

  MetalBuffer<MetalSplitFindTask> tasks_buf_;
  MetalBuffer<int8_t> is_feature_used_buf_;
  MetalBuffer<MetalSplitResult> per_task_result_buf_;
  MetalBuffer<MetalSplitResult> per_leaf_best_buf_;
  MetalBuffer<float> histogram_input_buf_;

  void* find_best_split_pso_;
};

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_METAL_BEST_SPLIT_FINDER_HPP_
