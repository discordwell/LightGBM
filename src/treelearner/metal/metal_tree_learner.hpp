/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_TREE_LEARNER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_TREE_LEARNER_HPP_

#include "../serial_tree_learner.h"

#ifdef LGBM_USE_METAL

#include "metal_leaf_splits.hpp"
#include "metal_histogram_constructor.hpp"
#include "metal_best_split_finder.hpp"

#include <memory>
#include <vector>

namespace LightGBM {

class MetalSingleGPUTreeLearner : public SerialTreeLearner {
 public:
  explicit MetalSingleGPUTreeLearner(const Config* config);
  ~MetalSingleGPUTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  Tree* Train(const score_t* gradients, const score_t* hessians, bool is_first_tree) override;
  void ResetTrainingData(const Dataset* train_data, bool is_constant_hessian) override;
  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override;

 protected:
  void BeforeTrain() override;

  std::unique_ptr<MetalLeafSplits> smaller_leaf_splits_;
  std::unique_ptr<MetalLeafSplits> larger_leaf_splits_;
  std::unique_ptr<MetalHistogramConstructor> histogram_constructor_;
  std::unique_ptr<MetalBestSplitFinder> best_split_finder_;

  std::vector<int> leaf_best_split_feature_;
  std::vector<uint32_t> leaf_best_split_threshold_;
  std::vector<uint8_t> leaf_best_split_default_left_;
  std::vector<data_size_t> leaf_num_data_;
  std::vector<data_size_t> leaf_data_start_;
  std::vector<double> leaf_sum_gradients_;
  std::vector<double> leaf_sum_hessians_;

  int smaller_leaf_index_;
  int larger_leaf_index_;
  int best_leaf_index_;
  bool has_categorical_feature_;
  int num_threads_;
};

}  // namespace LightGBM

#else

namespace LightGBM {

class MetalSingleGPUTreeLearner : public SerialTreeLearner {
 public:
  explicit MetalSingleGPUTreeLearner(const Config* config) : SerialTreeLearner(config) {
    Log::Fatal("Metal Tree Learner was not enabled in this build.\n"
               "Please recompile with CMake option -DLGBM_USE_METAL=1");
  }
};

}  // namespace LightGBM

#endif
#endif
