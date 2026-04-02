/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_SRC_TREELEARNER_METAL_METAL_LEAF_SPLITS_HPP_
#define LIGHTGBM_SRC_TREELEARNER_METAL_METAL_LEAF_SPLITS_HPP_

#ifdef LGBM_USE_METAL

#include "metal_utils.hpp"

#include <LightGBM/bin.h>
#include <LightGBM/meta.h>

namespace LightGBM {

/*!
 * \brief POD struct that lives in a MetalBuffer (StorageModeShared) so both
 *        CPU and GPU can read/write it without explicit copies.
 *
 *        Compared with CUDALeafSplitsStruct the pointer members are replaced
 *        by offsets because Metal shaders cannot chase host pointers.
 */
struct MetalLeafSplitsStruct {
  int leaf_index;
  double sum_of_gradients;
  double sum_of_hessians;
  data_size_t num_data_in_leaf;
  double gain;
  double leaf_value;
  /*! Byte offset into the data-partition index buffer. */
  data_size_t data_indices_offset;
  /*! Element offset into the histogram buffer. */
  int64_t hist_offset;
};

/*!
 * \brief Manages per-leaf statistics for the Metal tree learner.
 *
 *        Root-node sums are computed on the CPU with OpenMP parallel
 *        reduction over the shared-memory gradient/hessian arrays.  This is
 *        fast (~0.5 ms for 10 M rows on Apple Silicon) and avoids a GPU
 *        kernel launch + synchronisation round-trip.
 */
class MetalLeafSplits {
 public:
  explicit MetalLeafSplits(data_size_t num_data);
  ~MetalLeafSplits();

  /*!
   * \brief Initialise for the root leaf.  Sums all gradients and hessians,
   *        computes gain and leaf value, and fills the shared struct.
   */
  void Init(const score_t* gradients, const score_t* hessians,
            const data_size_t* data_indices, data_size_t num_data,
            hist_t* hist_data, double lambda_l1, double lambda_l2);

  /*! \brief Reset the struct to represent an empty (invalid) leaf. */
  void InitValues();

  MetalLeafSplitsStruct* GetStruct() { return leaf_struct_.data(); }
  const MetalLeafSplitsStruct* GetStruct() const { return leaf_struct_.data(); }

  /*! \brief Underlying Metal buffer for binding to compute kernels. */
  void* GetMTLBuffer() { return leaf_struct_.GetMTLBuffer(); }

 private:
  static double ThresholdL1(double s, double l1);
  static double CalculateSplittedLeafOutput(double sum_gradients,
                                            double sum_hessians,
                                            double l1, double l2);
  static double GetLeafGain(double sum_gradients, double sum_hessians,
                            double l1, double l2);

  MetalBuffer<MetalLeafSplitsStruct> leaf_struct_;
  data_size_t num_data_;
};

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
#endif  // LIGHTGBM_SRC_TREELEARNER_METAL_METAL_LEAF_SPLITS_HPP_
