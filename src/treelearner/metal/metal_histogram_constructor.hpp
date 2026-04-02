/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_HISTOGRAM_CONSTRUCTOR_HPP_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_HISTOGRAM_CONSTRUCTOR_HPP_

#ifdef LGBM_USE_METAL

#include "metal_utils.hpp"
#include "metal_leaf_splits.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>
#include <LightGBM/train_share_states.h>

#include <memory>
#include <vector>

namespace LightGBM {

class MetalHistogramConstructor {
 public:
  MetalHistogramConstructor(const Dataset* train_data,
                            int num_leaves,
                            int num_threads,
                            const std::vector<uint32_t>& feature_hist_offsets,
                            int min_data_in_leaf,
                            double min_sum_hessian_in_leaf);

  ~MetalHistogramConstructor();

  void Init(const Dataset* train_data, TrainingShareStates* share_state);

  void BeforeTrain(const score_t* gradients, const score_t* hessians);

  void ConstructHistogramForLeaf(
      const MetalLeafSplitsStruct* smaller_leaf,
      const MetalLeafSplitsStruct* larger_leaf,
      data_size_t num_data_in_smaller_leaf,
      data_size_t num_data_in_larger_leaf,
      double sum_hessians_in_smaller_leaf,
      double sum_hessians_in_larger_leaf);

  void SubtractHistogramForLeaf(
      const MetalLeafSplitsStruct* smaller_leaf,
      const MetalLeafSplitsStruct* larger_leaf);

  void ResetTrainingData(const Dataset* train_data,
                         TrainingShareStates* share_states);

  hist_t* hist_data() { return hist_buf_.data(); }

  const hist_t* hist_data() const { return hist_buf_.data(); }

  void* GetHistMTLBuffer() { return hist_buf_.GetMTLBuffer(); }

  int num_total_bin() const { return num_total_bin_; }

 private:
  void InitFeatureMetaInfo(const Dataset* train_data,
                           const std::vector<uint32_t>& feature_hist_offsets);

  void InitRowData(const Dataset* train_data, TrainingShareStates* share_state);

  data_size_t num_data_;
  int num_features_;
  int num_leaves_;
  int num_threads_;
  int num_total_bin_;

  std::vector<uint32_t> feature_num_bins_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint32_t> feature_most_freq_bins_;

  int min_data_in_leaf_;
  double min_sum_hessian_in_leaf_;

  std::vector<int> need_fix_histogram_features_;
  std::vector<uint32_t> need_fix_histogram_features_num_bin_aligned_;

  MetalBuffer<hist_t> hist_buf_;
  MetalBuffer<uint32_t> feature_num_bins_buf_;
  MetalBuffer<uint32_t> feature_hist_offsets_buf_;
  MetalBuffer<uint32_t> feature_most_freq_bins_buf_;

  MetalBuffer<uint8_t> row_bin_data_;
  MetalBuffer<uint32_t> row_ptr_offsets_;
  int row_data_bit_type_;

  const score_t* gradients_;
  const score_t* hessians_;

  void* histogram_pso_;
  void* subtract_pso_;
};

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_METAL_HISTOGRAM_CONSTRUCTOR_HPP_
