/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef LGBM_USE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_leaf_splits.hpp"

#include "../data_partition.hpp"
#include "../leaf_splits.hpp"

#include <cmath>

namespace LightGBM {

// ---------------------------------------------------------------------------
//  Construction / destruction
// ---------------------------------------------------------------------------

MetalLeafSplits::MetalLeafSplits(data_size_t num_data, size_t num_slots)
    : leaf_struct_(std::max<size_t>(static_cast<size_t>(1), num_slots)),
      num_data_(num_data),
      num_slots_(std::max<size_t>(static_cast<size_t>(1), num_slots)) {
  Reset();
}

MetalLeafSplits::~MetalLeafSplits() {}

void MetalLeafSplits::ResizeSlots(size_t num_slots) {
  num_slots_ = std::max<size_t>(static_cast<size_t>(1), num_slots);
  leaf_struct_.Resize(num_slots_);
  Reset();
}

void MetalLeafSplits::Reset() {
  for (size_t slot = 0; slot < num_slots_; ++slot) {
    InitValues(slot);
  }
}

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

void MetalLeafSplits::SetLeafState(size_t slot,
                                   int leaf_index,
                                   double sum_gradients,
                                   double sum_hessians,
                                   data_size_t num_data_in_leaf,
                                   double leaf_value,
                                   data_size_t data_indices_offset,
                                   int64_t hist_offset) {
  CHECK_LT(slot, num_slots_);
  MetalLeafSplitsStruct* s = GetStruct(slot);
  s->leaf_index = leaf_index;
  s->sum_of_gradients = sum_gradients;
  s->sum_of_hessians = sum_hessians;
  s->num_data_in_leaf = num_data_in_leaf;
  s->gain = GetLeafGain(sum_gradients, sum_hessians, 0.0, 0.0);
  s->leaf_value = leaf_value;
  s->data_indices_offset = data_indices_offset;
  s->hist_offset = hist_offset;
}

void MetalLeafSplits::InitValues() {
  InitValues(0);
}

void MetalLeafSplits::InitValues(size_t slot) {
  CHECK_LT(slot, num_slots_);
  MetalLeafSplitsStruct* s = GetStruct(slot);
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

  SetLeafState(0, 0, sum_grad, sum_hess, num_data,
               CalculateSplittedLeafOutput(sum_grad, sum_hess, lambda_l1,
                                           lambda_l2),
               0, 0);
  GetStruct(0)->gain = GetLeafGain(sum_grad, sum_hess, lambda_l1, lambda_l2);
}

void MetalLeafSplits::SyncLeaf(size_t slot,
                               const LeafSplits* leaf_splits,
                               const DataPartition* data_partition,
                               int64_t hist_offset) {
  if (leaf_splits == nullptr || data_partition == nullptr ||
      leaf_splits->leaf_index() < 0) {
    InitValues(slot);
    return;
  }
  const int leaf_index = leaf_splits->leaf_index();
  SetLeafState(slot, leaf_index,
               leaf_splits->sum_gradients(),
               leaf_splits->sum_hessians(),
               leaf_splits->num_data_in_leaf(),
               leaf_splits->weight(),
               data_partition->leaf_begin(leaf_index),
               hist_offset);
}

}  // namespace LightGBM

#endif  // LGBM_USE_METAL
