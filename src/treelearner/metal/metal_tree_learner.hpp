/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_TREE_LEARNER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_TREE_LEARNER_HPP_

#include "../serial_tree_learner.h"

#ifdef LGBM_USE_METAL

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace LightGBM {

class MetalBestSplitFinder;
class MetalLeafSplits;
struct MetalLeafSplitsStruct;

/*!
 * \brief Metal GPU-accelerated tree learner for Apple Silicon.
 *
 * Current v1 scope is performance-first and intentionally narrow: dense
 * numerical data only, serial GBDT only, and max_bin <= 256. Histogram
 * construction is dispatched to Metal; the serial CPU loop still owns tree
 * mutation and data partitioning by default, and remains the default
 * split-search path until the experimental Metal split finder / partitioner
 * are fast enough to help end-to-end wall time.
 *
 * Two kernel strategies, auto-selected based on feature count:
 *  - Row-parallel (wide datasets): one thread per row, all features per thread.
 *    Reads each gradient once instead of N_features times.
 *  - Column-grouped (narrow datasets): one threadgroup per feature group,
 *    fast threadgroup-local histogram with CAS-loop atomics.
 */
class MetalSingleGPUTreeLearner : public SerialTreeLearner {
 public:
  explicit MetalSingleGPUTreeLearner(const Config* config);
  ~MetalSingleGPUTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  Tree* Train(const score_t* gradients, const score_t* hessians, bool is_first_tree) override;
  void ResetTrainingData(const Dataset* train_data, bool is_constant_hessian) override;
  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients,
                         const score_t* hessians) const override;
  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                         const score_t* gradients, const score_t* hessians) const override;

 protected:
  void BeforeTrain() override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;
  void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used,
                                    bool use_subtract, const Tree* tree) override;
  void Split(Tree* tree, int best_leaf, int* left_leaf,
             int* right_leaf) override;

 private:
  typedef float gpu_hist_t;

  /*! \brief Initialize Metal device, load metallib, create pipeline */
  void InitMetal();
  void ValidateTrainingScope(const Dataset* train_data) const;

  /*! \brief Allocate Metal buffers for gradient/hessian/indices */
  void AllocateMetalBuffers();
  void ResetMetalLeafStateTable();
  void SyncMetalActiveLeafState();
  void SyncMetalLeafState(int leaf_index, const LeafSplits* leaf_splits);
  void UpdateMetalLeafState(int leaf_index, double sum_gradients,
                            double sum_hessians, data_size_t num_data_in_leaf,
                            double leaf_value);
  double GetMetalParentOutput(const Tree* tree,
                              const MetalLeafSplitsStruct* leaf_state) const;
  const MetalLeafSplitsStruct* GetActiveMetalLeafState(size_t slot) const;
  const MetalLeafSplitsStruct* FindActiveMetalLeafState(int leaf_index) const;

  // Metal objects (opaque pointers to Objective-C types)
  void* metal_device_ = nullptr;
  void* metal_queue_ = nullptr;
  void* metal_library_ = nullptr;
  void* histogram_pipeline_ = nullptr;         // column-grouped kernel
  void* histogram_row_pipeline_ = nullptr;     // gathered sub-histogram kernel
  void* reduction_pipeline_ = nullptr;         // sub-histogram reduction kernel
  void* gather_pipeline_ = nullptr;            // gradient / bin reorder kernel
  void* packed_histogram_pipeline_ = nullptr;  // packed-tuple histogram kernel
  void* packed_reduction_pipeline_ = nullptr;  // packed-tuple reduction kernel
  void* packed_gather_pipeline_ = nullptr;     // packed-tuple gather kernel
  void* histogram_subtract_pipeline_ = nullptr;  // cached parent-smaller subtraction
  void* partition_pipeline_ = nullptr;         // numerical partition kernel

  // Metal buffers
  void* gradients_buffer_ = nullptr;
  void* hessians_buffer_ = nullptr;
  void* ordered_grad_buffer_ = nullptr;       // pre-gathered for row-parallel
  void* ordered_hess_buffer_ = nullptr;       // pre-gathered for row-parallel
  void* ordered_bins_buffer_ = nullptr;       // pre-gathered bin data
  void* ordered_packed_bins_buffer_ = nullptr;  // pre-gathered uchar4 tuples
  void* subhist_buffer_ = nullptr;            // gathered sub-histogram scratch
  void* data_indices_buffer_ = nullptr;
  void* partition_output_buffer_ = nullptr;
  void* partition_counts_buffer_ = nullptr;
  void* histogram_output_buffer_ = nullptr;
  void* leaf_hist_cache_buffer_ = nullptr;   // [num_leaves × total_bins × 2] float cache
  void* leaf_hist_parent_buffer_ = nullptr;  // one histogram scratch when parent slot is reused

  // Bin data (allocated on first use)
  void* bin_data_col_buffer_ = nullptr;   // column-major [groups × rows]
  void* bin_data_packed_buffer_ = nullptr;  // tuple-major [tuples × rows] of uchar4
  void* bin_data_row_buffer_ = nullptr;   // unused
  void* dense_group_map_buffer_ = nullptr;  // [tuples × 4] dense group ids
  void* group_offsets_buffer_ = nullptr;
  bool bin_data_packed_ = false;
  bool use_row_parallel_ = false;         // auto-selected based on feature count
  std::vector<uint32_t> group_bin_offsets_;
  std::vector<uint32_t> dense_group_map_;
  std::unique_ptr<MetalBestSplitFinder> best_split_finder_;
  std::unique_ptr<MetalLeafSplits> metal_leaf_splits_;

  // Feature layout
  int num_feature_groups_;
  int num_dense_feature_groups_;
  int num_dense_feature_tuples_ = 0;
  int max_num_bin_;
  size_t leaf_hist_num_items_ = 0;

  data_size_t PartitionLeafOnGPU(int leaf, int inner_feature_index,
                                 uint32_t threshold, bool default_left);
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
