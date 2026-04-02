/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_TREE_LEARNER_HPP_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_TREE_LEARNER_HPP_

#include "../serial_tree_learner.h"

#ifdef LGBM_USE_METAL

#include <memory>
#include <vector>

namespace LightGBM {

/*!
 * \brief Metal GPU-accelerated tree learner for Apple Silicon.
 *
 * Follows the GPUTreeLearner (OpenCL) pattern: overrides only histogram
 * construction to dispatch a Metal compute kernel. Split finding, data
 * partitioning, and all other logic use SerialTreeLearner's CPU code.
 *
 * Unlike the previous v1 approach (which had a custom training loop),
 * this design reuses the proven CPU infrastructure and only accelerates
 * the bottleneck (histogram construction).
 */
class MetalSingleGPUTreeLearner : public SerialTreeLearner {
 public:
  explicit MetalSingleGPUTreeLearner(const Config* config);
  ~MetalSingleGPUTreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  Tree* Train(const score_t* gradients, const score_t* hessians, bool is_first_tree) override;
  void ResetTrainingData(const Dataset* train_data, bool is_constant_hessian) override;

 protected:
  void BeforeTrain() override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;

 private:
  /*! \brief 4-byte feature tuple used by GPU kernel (matches OpenCL GPUTreeLearner) */
  struct Feature4 {
    uint8_t s[4];
  };

  typedef float gpu_hist_t;

  /*! \brief Initialize Metal device, load metallib, create pipeline */
  void InitMetal();

  /*! \brief Pack feature data into Feature4 format for GPU */
  void AllocateMetalBuffers();

  /*! \brief Build GPU histogram for given leaf data */
  void BuildMetalHistogram(data_size_t num_data, const data_size_t* data_indices);

  /*! \brief Wait for GPU and copy histogram results */
  void WaitAndGetHistograms(hist_t* histograms);

  // Metal objects (opaque pointers to Objective-C types)
  void* metal_device_ = nullptr;
  void* metal_queue_ = nullptr;
  void* metal_library_ = nullptr;
  void* histogram_pipeline_ = nullptr;
  void* pending_command_buffer_ = nullptr;

  // Metal buffers
  void* features_buffer_ = nullptr;    // Feature4 packed data
  void* gradients_buffer_ = nullptr;   // Cached gradient copy
  void* hessians_buffer_ = nullptr;    // Cached hessian copy
  void* data_indices_buffer_ = nullptr;
  void* histogram_output_buffer_ = nullptr;  // float histogram output

  // Feature layout
  int num_feature_groups_;
  int num_dense_feature_groups_;
  int num_dense_feature4_;
  int dword_features_;
  int device_bin_size_;
  size_t hist_bin_entry_sz_;
  std::vector<int> dense_feature_group_map_;
  std::vector<int> sparse_feature_group_map_;
  std::vector<int> device_bin_mults_;
  std::vector<char> feature_masks_;
  int max_num_bin_;
  std::string kernel_name_;

  // GPU histogram data
  void* bin_data_buffer_ = nullptr;     // Packed row-major bin data
  void* group_offsets_buffer_ = nullptr;// Group bin boundary offsets
  bool bin_data_packed_ = false;
  std::vector<uint32_t> group_bin_offsets_;
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
