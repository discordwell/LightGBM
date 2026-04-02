/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_BEST_SPLIT_FINDER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_BEST_SPLIT_FINDER_HPP_

#ifdef LGBM_USE_METAL

#include "metal_utils.hpp"
#include "metal_leaf_splits.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/config.h>
#include <LightGBM/dataset.h>

#include <vector>

#include "../split_info.hpp"

namespace LightGBM {

struct MetalSplitFindTask {
  int inner_feature_index;
  int8_t reverse;
  int8_t skip_default_bin;
  int8_t na_as_missing;
  int8_t assume_out_default_left;
  int8_t is_categorical;
  int8_t is_one_hot;
  uint32_t hist_offset;
  uint8_t mfb_offset;
  uint32_t num_bin;
  uint32_t default_bin;
};

struct MetalSplitResult {
  double gain;
  int feature;
  uint32_t threshold;
  int8_t default_left;
  double left_sum_gradient;
  double left_sum_hessian;
  data_size_t left_count;
  double right_sum_gradient;
  double right_sum_hessian;
  data_size_t right_count;
  double left_value;
  double right_value;
  int8_t found;
};

class MetalBestSplitFinder {
 public:
  MetalBestSplitFinder(const hist_t* hist_data,
                       const Dataset* train_data,
                       const std::vector<uint32_t>& feature_hist_offsets,
                       const Config* config);

  ~MetalBestSplitFinder();

  void Init();

  void BeforeTrain(const std::vector<int8_t>& is_feature_used_bytree);

  void FindBestSplitsForLeaf(
      const MetalLeafSplitsStruct* smaller_leaf,
      const MetalLeafSplitsStruct* larger_leaf,
      int smaller_leaf_index,
      int larger_leaf_index,
      data_size_t num_data_in_smaller_leaf,
      data_size_t num_data_in_larger_leaf,
      double sum_hessians_in_smaller_leaf,
      double sum_hessians_in_larger_leaf);

  void FindBestFromAllSplits(
      int cur_num_leaves,
      int smaller_leaf_index,
      int larger_leaf_index,
      int* best_leaf_index,
      SplitInfo* best_split);

  void ResetTrainingData(const hist_t* hist_data,
                         const Dataset* train_data,
                         const std::vector<uint32_t>& feature_hist_offsets);

  void ResetConfig(const Config* config, const hist_t* hist_data);

 private:
  void InitFeatureMetaInfo(const Dataset* train_data);
  void InitTasks();

  int num_features_;
  int num_leaves_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_default_bins_;
  std::vector<uint32_t> feature_num_bins_;
  std::vector<MissingType> feature_missing_type_;
  std::vector<int8_t> is_categorical_;

  double lambda_l1_;
  double lambda_l2_;
  data_size_t min_data_in_leaf_;
  double min_sum_hessian_in_leaf_;
  double min_gain_to_split_;
  double cat_smooth_;
  double cat_l2_;
  int max_cat_threshold_;
  int min_data_per_group_;
  int max_cat_to_onehot_;

  bool has_categorical_feature_;
  int max_num_categorical_bin_;

  int num_tasks_;
  std::vector<MetalSplitFindTask> split_find_tasks_;

  MetalBuffer<MetalSplitFindTask> tasks_buf_;
  MetalBuffer<int8_t> is_feature_used_buf_;
  MetalBuffer<MetalSplitResult> per_task_result_buf_;
  MetalBuffer<MetalSplitResult> per_leaf_best_buf_;

  const hist_t* hist_data_;

  void* find_best_split_pso_;
  void* sync_best_split_pso_;
  void* find_best_from_all_pso_;
};

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_METAL_BEST_SPLIT_FINDER_HPP_
