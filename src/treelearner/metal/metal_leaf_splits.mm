/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef LGBM_USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_leaf_splits.hpp"

#include <cmath>

namespace LightGBM {

// ---------------------------------------------------------------------------
//  Construction / destruction
// ---------------------------------------------------------------------------

MetalLeafSplits::MetalLeafSplits(data_size_t num_data)
    : leaf_struct_(1), num_data_(num_data) {
  InitValues();
}

MetalLeafSplits::~MetalLeafSplits() {}

// ---------------------------------------------------------------------------
//  Leaf math helpers (mirror CUDALeafSplits device functions, CPU-side)
// ---------------------------------------------------------------------------

double MetalLeafSplits::ThresholdL1(double s, double l1) {
  const double reg_s = std::fmax(0.0, std::fabs(s) - l1);
  return (s >= 0.0) ? reg_s : -reg_s;
}

double MetalLeafSplits::CalculateSplittedLeafOutput(double sum_gradients,
                                                    double sum_hessians,
                                                    double l1, double l2) {
  if (l1 > 0.0) {
    return -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
  }
  return -sum_gradients / (sum_hessians + l2);
}

double MetalLeafSplits::GetLeafGain(double sum_gradients,
                                    double sum_hessians,
                                    double l1, double l2) {
  if (l1 > 0.0) {
    const double sg_l1 = ThresholdL1(sum_gradients, l1);
    return (sg_l1 * sg_l1) / (sum_hessians + l2);
  }
  return (sum_gradients * sum_gradients) / (sum_hessians + l2);
}

// ---------------------------------------------------------------------------
//  InitValues — mark the leaf as empty / invalid
// ---------------------------------------------------------------------------

void MetalLeafSplits::InitValues() {
  MetalLeafSplitsStruct* s = leaf_struct_.data();
  s->leaf_index = -1;
  s->sum_of_gradients = 0.0;
  s->sum_of_hessians = 0.0;
  s->num_data_in_leaf = 0;
  s->gain = 0.0;
  s->leaf_value = 0.0;
  s->data_indices_offset = 0;
  s->hist_offset = 0;
}

// ---------------------------------------------------------------------------
//  Init — root leaf initialisation via CPU parallel reduction
// ---------------------------------------------------------------------------

void MetalLeafSplits::Init(const score_t* gradients,
                           const score_t* hessians,
                           const data_size_t* data_indices,
                           data_size_t num_data,
                           hist_t* hist_data,
                           double lambda_l1,
                           double lambda_l2) {
  num_data_ = num_data;

  double sum_grad = 0.0;
  double sum_hess = 0.0;

  if (data_indices != nullptr) {
    #pragma omp parallel for schedule(static) reduction(+:sum_grad, sum_hess)
    for (data_size_t i = 0; i < num_data; ++i) {
      const data_size_t idx = data_indices[i];
      sum_grad += static_cast<double>(gradients[idx]);
      sum_hess += static_cast<double>(hessians[idx]);
    }
  } else {
    #pragma omp parallel for schedule(static) reduction(+:sum_grad, sum_hess)
    for (data_size_t i = 0; i < num_data; ++i) {
      sum_grad += static_cast<double>(gradients[i]);
      sum_hess += static_cast<double>(hessians[i]);
    }
  }

  MetalLeafSplitsStruct* s = leaf_struct_.data();
  s->leaf_index = 0;
  s->sum_of_gradients = sum_grad;
  s->sum_of_hessians = sum_hess;
  s->num_data_in_leaf = num_data;
  s->gain = GetLeafGain(sum_grad, sum_hess, lambda_l1, lambda_l2);
  s->leaf_value = CalculateSplittedLeafOutput(sum_grad, sum_hess,
                                              lambda_l1, lambda_l2);
  s->data_indices_offset = 0;
  s->hist_offset = 0;
}

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
