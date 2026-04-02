/*!
 * Copyright (c) 2024 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2024 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * Metal GPU kernels for LightGBM tree learner.
 * Compiled to lib_lightgbm.metallib at build time.
 *
 * Ported from CUDA kernels:
 *   - cuda_histogram_constructor.cu
 *   - cuda_best_split_finder.cu
 *   - cuda_leaf_splits.cu
 */

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant float kEpsilon = 1e-15f;
constant float kMinScore = -INFINITY;
constant uint SIMD_SIZE = 32;  // Apple Silicon SIMD width

// Threadgroup histogram max entries.
// Metal limit: 32KB threadgroup memory. With float pairs (8 bytes each),
// 32768 / 8 = 4096 entries max. We use 3072 entries (24KB) to leave
// headroom for other threadgroup variables.
constant uint MAX_SHARED_HIST_ENTRIES = 3072;

// ---------------------------------------------------------------------------
// Struct definitions
// ---------------------------------------------------------------------------

struct GradientPair {
    float grad;
    float hess;
};

struct SplitInfo {
    float gain;
    uint  feature_index;
    uint  threshold;
    uint  left_count;
    float left_sum_gradients;
    float left_sum_hessians;
    float right_sum_gradients;
    float right_sum_hessians;
    float left_value;
    float left_gain;
    float right_value;
    float right_gain;
    uchar default_left;
    uchar is_valid;
    int   inner_feature_index;
};

struct LeafSplitsStruct {
    int   leaf_index;
    float sum_of_gradients;
    float sum_of_hessians;
    uint  num_data_in_leaf;
    float gain;
    float leaf_value;
    uint  data_indices_offset;   // offset into global data_indices buffer
    uint  hist_offset;           // offset into global histogram buffer
};

struct SplitFindTask {
    int   inner_feature_index;
    int   reverse;               // bool as int for Metal struct alignment
    int   skip_default_bin;
    int   na_as_missing;
    int   assume_out_default_left;
    uint  hist_offset;
    uint  mfb_offset;
    uint  num_bin;
    uint  default_bin;
};

// ---------------------------------------------------------------------------
// Helper functions: CAS-loop float atomics (M1/M2 compatible)
// ---------------------------------------------------------------------------

/// Atomic float add for threadgroup memory via compare-and-swap loop.
inline void atomic_add_float(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    float current_val;
    uint desired;
    bool exchanged = false;
    while (!exchanged) {
        current_val = as_type<float>(expected);
        desired = as_type<uint>(current_val + val);
        exchanged = atomic_compare_exchange_weak_explicit(
            addr, &expected, desired,
            memory_order_relaxed, memory_order_relaxed);
    }
}

/// Atomic float add for device memory via compare-and-swap loop.
inline void atomic_add_float_device(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    float current_val;
    uint desired;
    bool exchanged = false;
    while (!exchanged) {
        current_val = as_type<float>(expected);
        desired = as_type<uint>(current_val + val);
        exchanged = atomic_compare_exchange_weak_explicit(
            addr, &expected, desired,
            memory_order_relaxed, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Helper functions: gain/output calculations
// ---------------------------------------------------------------------------

/// L1 thresholding: soft-threshold gradient by l1 penalty.
inline float ThresholdL1(float s, float l1) {
    float reg_s = max(0.0f, abs(s) - l1);
    return (s >= 0.0f) ? reg_s : -reg_s;
}

/// Compute leaf gain = g^2 / (h + lambda_l2), with optional L1.
inline float GetLeafGain(float sum_gradients, float sum_hessians,
                         float lambda_l1, float lambda_l2) {
    if (lambda_l1 > 0.0f) {
        float sg_l1 = ThresholdL1(sum_gradients, lambda_l1);
        return (sg_l1 * sg_l1) / (sum_hessians + lambda_l2);
    } else {
        return (sum_gradients * sum_gradients) / (sum_hessians + lambda_l2);
    }
}

/// Compute optimal leaf weight (output value).
inline float CalculateLeafOutput(float sum_gradients, float sum_hessians,
                                 float lambda_l1, float lambda_l2) {
    if (lambda_l1 > 0.0f) {
        return -ThresholdL1(sum_gradients, lambda_l1) / (sum_hessians + lambda_l2);
    } else {
        return -sum_gradients / (sum_hessians + lambda_l2);
    }
}

/// Compute leaf gain given a known output value.
inline float GetLeafGainGivenOutput(float sum_gradients, float sum_hessians,
                                    float lambda_l1, float lambda_l2, float output) {
    if (lambda_l1 > 0.0f) {
        float sg_l1 = ThresholdL1(sum_gradients, lambda_l1);
        return -(2.0f * sg_l1 * output + (sum_hessians + lambda_l2) * output * output);
    } else {
        return -(2.0f * sum_gradients * output + (sum_hessians + lambda_l2) * output * output);
    }
}

/// Compute split gain: sum of left and right leaf gains.
inline float GetSplitGain(float sum_left_gradients, float sum_left_hessians,
                          float sum_right_gradients, float sum_right_hessians,
                          float lambda_l1, float lambda_l2) {
    return GetLeafGain(sum_left_gradients, sum_left_hessians, lambda_l1, lambda_l2) +
           GetLeafGain(sum_right_gradients, sum_right_hessians, lambda_l1, lambda_l2);
}

// ---------------------------------------------------------------------------
// Helper functions: SIMD reductions
// ---------------------------------------------------------------------------

/// Sum-reduce a float value across the SIMD group (warp).
inline float simd_reduce_sum_float(float val) {
    return simd_sum(val);
}

/// Max-reduce a float value across the SIMD group, returning both
/// the max value and the lane index that held it.
/// Uses simd_shuffle_down to perform a tree reduction.
inline void simd_reduce_max_with_index(float val, uint idx,
                                       threadgroup float* shared_gain,
                                       threadgroup uint* shared_idx,
                                       uint simd_lane, uint simd_group_id,
                                       uint simd_groups_per_tg) {
    // Intra-SIMD reduction via shuffle
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(val, offset);
        uint  other_idx = simd_shuffle_down(idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    // Lane 0 of each SIMD group writes to shared memory
    if (simd_lane == 0) {
        shared_gain[simd_group_id] = val;
        shared_idx[simd_group_id]  = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // First SIMD group reduces across all SIMD groups
    if (simd_group_id == 0) {
        val = (simd_lane < simd_groups_per_tg) ? shared_gain[simd_lane] : kMinScore;
        idx = (simd_lane < simd_groups_per_tg) ? shared_idx[simd_lane]  : 0;
        for (uint offset = SIMD_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val = simd_shuffle_down(val, offset);
            uint  other_idx = simd_shuffle_down(idx, offset);
            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }
        if (simd_lane == 0) {
            shared_gain[0] = val;
            shared_idx[0]  = idx;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/// Inclusive prefix sum across the threadgroup using SIMD primitives.
/// Works for threadgroup sizes up to SIMD_SIZE * SIMD_SIZE (1024).
/// val:          the per-thread value to scan
/// shared_buf:   threadgroup buffer of size >= number of SIMD groups
/// tid:          thread index in threadgroup
/// tg_size:      threads per threadgroup
inline float threadgroup_prefix_sum(float val,
                                    threadgroup float* shared_buf,
                                    uint tid, uint tg_size) {
    // Step 1: inclusive prefix sum within each SIMD group
    float scanned = simd_prefix_inclusive_sum(val);

    uint simd_group_id = tid / SIMD_SIZE;
    uint simd_lane     = tid % SIMD_SIZE;
    uint num_simds     = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;

    // Step 2: last lane of each SIMD group writes its total to shared memory
    if (simd_lane == SIMD_SIZE - 1 || tid == tg_size - 1) {
        shared_buf[simd_group_id] = scanned;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: first SIMD group scans the per-group totals
    if (simd_group_id == 0) {
        float group_val = (simd_lane < num_simds) ? shared_buf[simd_lane] : 0.0f;
        float group_scanned = simd_prefix_inclusive_sum(group_val);
        if (simd_lane < num_simds) {
            shared_buf[simd_lane] = group_scanned;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: add prefix from previous SIMD groups
    if (simd_group_id > 0) {
        scanned += shared_buf[simd_group_id - 1];
    }
    return scanned;
}

/// Sum-reduce a float across the entire threadgroup.
/// shared_buf must have >= num_simd_groups entries.
inline float threadgroup_reduce_sum(float val,
                                    threadgroup float* shared_buf,
                                    uint tid, uint tg_size) {
    float local_sum = simd_sum(val);
    uint simd_group_id = tid / SIMD_SIZE;
    uint simd_lane     = tid % SIMD_SIZE;
    uint num_simds     = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;

    if (simd_lane == 0) {
        shared_buf[simd_group_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane < num_simds) ? shared_buf[simd_lane] : 0.0f;
        float total = simd_sum(v);
        if (simd_lane == 0) {
            shared_buf[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared_buf[0];
}


// ===========================================================================
// KERNEL 1: histogram_dense
//
// Construct gradient/hessian histograms for dense feature storage.
// Dispatch: threadgroups = (num_feature_partitions, num_data_blocks, 1)
//           threads_per_threadgroup = (num_columns_in_partition, block_y, 1)
//
// Simplified Metal port: one threadgroup per feature column.
// threadgroups.x = num_features, threads_per_threadgroup.x = 256
// Each thread iterates over a portion of the data rows in this leaf.
// ===========================================================================

// Simple histogram kernel matching host buffer layout:
//   buffer(0): float* gradients          [num_data]
//   buffer(1): float* hessians           [num_data]
//   buffer(2): uchar* row_bin_data       [num_data * num_features] row-major
//   buffer(3): uint*  feature_hist_offsets [num_features+1] cumulative bin offsets
//   buffer(4): uint*  feature_mfb         [num_features] most-frequent-bin per feature
//   buffer(5): float* hist_output         [num_total_bin * 2] interleaved grad/hess
//   buffer(6): HistogramParams struct     {num_data, num_features, num_total_bin, bit_type}
//   buffer(7): int    data_offset         start index into data partition
//   buffer(8): uint*  feature_num_bins    [num_features]
// Dispatch: one threadgroup per feature, threads iterate over rows.
struct HistogramParams {
    uint num_data;
    uint num_features;
    uint num_total_bin;
    uint bit_type;
};

kernel void histogram_dense(
    const device float*       gradients        [[buffer(0)]],
    const device float*       hessians         [[buffer(1)]],
    const device uchar*       row_bin_data     [[buffer(2)]],
    const device uint*        feature_hist_offsets [[buffer(3)]],
    const device uint*        feature_mfb      [[buffer(4)]],
    device float*             hist_output      [[buffer(5)]],
    constant HistogramParams& params           [[buffer(6)]],
    constant int&             data_offset      [[buffer(7)]],
    const device uint*        feature_num_bins [[buffer(8)]],
    uint tid       [[thread_position_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    uint group_id  [[threadgroup_position_in_grid]])
{
    const uint feature = group_id;
    if (feature >= params.num_features) return;

    const uint hist_start = feature_hist_offsets[feature];
    const uint nbins = feature_num_bins[feature];
    const uint mfb = feature_mfb[feature];

    // Threadgroup-local histogram (grad + hess per bin)
    threadgroup atomic_uint local_hist[MAX_SHARED_HIST_ENTRIES * 2];

    // Zero bins for this feature
    for (uint i = tid; i < nbins * 2; i += tg_size) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate gradients/hessians into local histogram
    const uint num_rows = params.num_data;
    const uint num_feat = params.num_features;
    for (uint r = tid; r < num_rows; r += tg_size) {
        const uint row_idx = r;  // data_offset is handled by host via partition indices
        const uint bin = uint(row_bin_data[row_idx * num_feat + feature]);
        if (bin == mfb) continue;  // skip most-frequent-bin (fixed on CPU later)

        const float g = gradients[row_idx];
        const float h = hessians[row_idx];
        atomic_add_float(&local_hist[bin * 2], g);
        atomic_add_float(&local_hist[bin * 2 + 1], h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global output (no inter-threadgroup contention
    // since each threadgroup handles a different feature)
    for (uint i = tid; i < nbins * 2; i += tg_size) {
        float val = as_type<float>(atomic_load_explicit(&local_hist[i], memory_order_relaxed));
        hist_output[(hist_start + i / 2) * 2 + (i % 2)] = val;
    }
}


// ===========================================================================
// KERNEL 2: histogram_subtract
//
// Compute larger leaf histogram by subtracting smaller from parent:
//   larger_hist[i] = parent_hist[i] - smaller_hist[i]
//
// Dispatch: threadgroups = ceil(num_total_bins * 2 / threads_per_tg)
//           threads_per_threadgroup = 256
// ===========================================================================

kernel void histogram_subtract(
    const device float*   parent_hist    [[buffer(0)]],
    const device float*   smaller_hist   [[buffer(1)]],
    device float*         larger_hist    [[buffer(2)]],
    const device uint&    num_total_bins [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    const uint num_items = num_total_bins << 1;  // grad + hess per bin
    if (tid < num_items) {
        larger_hist[tid] = parent_hist[tid] - smaller_hist[tid];
    }
}


// ===========================================================================
// KERNEL 3: histogram_fix_mfb (most-frequent-bin fix)
//
// After histogram construction, the most-frequent bin (MFB) was NOT
// accumulated (it was skipped to save bandwidth). Compute its value as:
//   mfb_grad = total_grad - sum_of_all_other_bins_grad
//   mfb_hess = total_hess - sum_of_all_other_bins_hess
//
// One threadgroup per feature. Each thread sums a range of bins,
// then a threadgroup reduction gives the total across all bins (excluding MFB).
//
// Dispatch: threadgroups = num_features
//           threads_per_threadgroup = 256
// ===========================================================================

kernel void histogram_fix_mfb(
    device float*              histogram       [[buffer(0)]],
    const device uint*         feature_hist_offsets [[buffer(1)]],
    const device uint*         feature_num_bins     [[buffer(2)]],
    const device uint*         feature_mfb_bins     [[buffer(3)]],
    const device float*        leaf_sum_gradients   [[buffer(4)]],
    const device float*        leaf_sum_hessians    [[buffer(5)]],
    const device uint&         num_features         [[buffer(6)]],
    uint group_id      [[threadgroup_position_in_grid]],
    uint tid_in_group  [[thread_position_in_threadgroup]],
    uint tg_size       [[threads_per_threadgroup]])
{
    if (group_id >= num_features) return;

    const uint hist_offset = feature_hist_offsets[group_id];
    const uint num_bins    = feature_num_bins[group_id];
    const uint mfb_bin     = feature_mfb_bins[group_id];
    const float total_grad = *leaf_sum_gradients;
    const float total_hess = *leaf_sum_hessians;

    device float* hist_ptr = histogram + (hist_offset << 1);

    // Each thread sums its portion of non-MFB bins
    float thread_sum_grad = 0.0f;
    float thread_sum_hess = 0.0f;
    for (uint bin = tid_in_group; bin < num_bins; bin += tg_size) {
        if (bin != mfb_bin) {
            thread_sum_grad += hist_ptr[bin << 1];
            thread_sum_hess += hist_ptr[(bin << 1) + 1];
        }
    }

    // Threadgroup reduction
    threadgroup float shared_buf[32];
    float sum_grad = threadgroup_reduce_sum(thread_sum_grad, shared_buf, tid_in_group, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_hess = threadgroup_reduce_sum(thread_sum_hess, shared_buf, tid_in_group, tg_size);

    // Thread 0 writes the MFB entry
    if (tid_in_group == 0) {
        hist_ptr[mfb_bin << 1]       = total_grad - sum_grad;
        hist_ptr[(mfb_bin << 1) + 1] = total_hess - sum_hess;
    }
}


// ===========================================================================
// KERNEL 4: find_best_split
//
// Port of FindBestSplitsForLeafKernelInner (forward scan variant).
// One threadgroup per feature (SplitFindTask). Each thread handles a subset
// of bins. Computes prefix sum of grad/hess, evaluates split gain at each
// bin boundary, tracks best gain, then reduces across the threadgroup.
//
// Dispatch: threadgroups = num_tasks (one per feature)
//           threads_per_threadgroup = 256 (or 1024 for >256 bins)
// ===========================================================================

kernel void find_best_split(
    // Per-feature task descriptions
    const device SplitFindTask* tasks             [[buffer(0)]],
    const device uint&          num_tasks         [[buffer(1)]],
    // Feature-usage mask (1 = used, 0 = skip)
    const device char*          is_feature_used   [[buffer(2)]],
    // Leaf information
    const device LeafSplitsStruct* leaf_splits    [[buffer(3)]],
    // Histogram buffer (global, all features)
    const device float*         histogram         [[buffer(4)]],
    // Regularization parameters
    const device float&         lambda_l1         [[buffer(5)]],
    const device float&         lambda_l2         [[buffer(6)]],
    const device float&         min_gain_to_split [[buffer(7)]],
    const device int&           min_data_in_leaf  [[buffer(8)]],
    const device float&         min_sum_hessian_in_leaf [[buffer(9)]],
    // Output: one SplitInfo per task
    device SplitInfo*           output_splits     [[buffer(10)]],
    // Threadgroup / thread identification
    uint group_id      [[threadgroup_position_in_grid]],
    uint tid_in_group  [[thread_position_in_threadgroup]],
    uint tg_size       [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (group_id >= num_tasks) return;

    const device SplitFindTask& task = tasks[group_id];
    device SplitInfo& out = output_splits[group_id];

    // Mark invalid by default
    out.is_valid = 0;

    if (!is_feature_used[task.inner_feature_index]) {
        return;
    }

    const float sum_gradients = leaf_splits->sum_of_gradients;
    const float sum_hessians  = leaf_splits->sum_of_hessians + 2.0f * kEpsilon;
    const uint  num_data      = leaf_splits->num_data_in_leaf;
    const float parent_gain   = leaf_splits->gain;
    const float cnt_factor    = float(num_data) / sum_hessians;
    const float min_gain_shift = parent_gain + min_gain_to_split;

    const uint hist_base = task.hist_offset << 1;
    const device float* feature_hist_ptr = histogram + hist_base;

    const uint feature_num_bin_minus_offset = task.num_bin - task.mfb_offset;

    // Threadgroup shared memory for reductions
    threadgroup float shared_gain[32];
    threadgroup uint  shared_idx[32];
    threadgroup float shared_prefix[32];

    // -----------------------------------------------------------------------
    // Phase 1: Load histogram bin values into registers
    // -----------------------------------------------------------------------
    float local_grad = 0.0f;
    float local_hess = 0.0f;

    if (!task.reverse) {
        // Forward scan
        if (task.na_as_missing && task.mfb_offset == 1) {
            // NA-as-missing with mfb_offset=1: bin 0 is the "NA" bin
            // Thread tid_in_group handles bin (tid_in_group) where bin 0
            // gets the remainder (total - sum_of_non_default).
            if (tid_in_group < task.num_bin && tid_in_group > 0) {
                uint bin_offset = (tid_in_group - 1) << 1;
                local_grad = feature_hist_ptr[bin_offset];
                local_hess = feature_hist_ptr[bin_offset + 1];
            }
        } else {
            bool skip = (task.skip_default_bin &&
                         (tid_in_group + task.mfb_offset) == task.default_bin);
            if (tid_in_group < feature_num_bin_minus_offset && !skip) {
                uint bin_offset = tid_in_group << 1;
                local_grad = feature_hist_ptr[bin_offset];
                local_hess = feature_hist_ptr[bin_offset + 1];
            }
        }
    } else {
        // Reverse scan
        bool skip = (tid_in_group >= uint(task.na_as_missing)) &&
                    task.skip_default_bin &&
                    ((task.num_bin - 1 - tid_in_group) == task.default_bin);
        if (tid_in_group >= uint(task.na_as_missing) &&
            tid_in_group < feature_num_bin_minus_offset && !skip) {
            uint read_index = feature_num_bin_minus_offset - 1 - tid_in_group;
            uint bin_offset = read_index << 1;
            local_grad = feature_hist_ptr[bin_offset];
            local_hess = feature_hist_ptr[bin_offset + 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 1b: For na_as_missing forward, compute NA bin from remainder
    // -----------------------------------------------------------------------
    if (!task.reverse && task.na_as_missing && task.mfb_offset == 1) {
        float sum_grad_non_default = threadgroup_reduce_sum(local_grad, shared_prefix,
                                                            tid_in_group, tg_size);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum_hess_non_default = threadgroup_reduce_sum(local_hess, shared_prefix,
                                                            tid_in_group, tg_size);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid_in_group == 0) {
            local_grad += (sum_gradients - sum_grad_non_default);
            local_hess += (sum_hessians - sum_hess_non_default);
        }
    }

    // Add epsilon to bin 0's hessian to avoid division by zero in prefix sum
    if (tid_in_group == 0) {
        local_hess += kEpsilon;
    }

    // -----------------------------------------------------------------------
    // Phase 2: Inclusive prefix sum of grad and hess
    // -----------------------------------------------------------------------
    float prefix_grad = threadgroup_prefix_sum(local_grad, shared_prefix,
                                               tid_in_group, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float prefix_hess = threadgroup_prefix_sum(local_hess, shared_prefix,
                                               tid_in_group, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 3: Evaluate split gain at each bin boundary
    // -----------------------------------------------------------------------
    float local_best_gain = kMinScore;
    bool  threshold_found = false;
    uint  threshold_value = 0;
    float best_left_grad  = 0.0f;
    float best_left_hess  = 0.0f;

    if (task.reverse) {
        // Reverse: prefix_grad/prefix_hess is the right side
        bool skip = (tid_in_group >= uint(task.na_as_missing)) &&
                    task.skip_default_bin &&
                    ((task.num_bin - 1 - tid_in_group) == task.default_bin);
        if (tid_in_group >= uint(task.na_as_missing) &&
            tid_in_group <= task.num_bin - 2 && !skip) {
            float sum_right_gradient = prefix_grad;
            float sum_right_hessian  = prefix_hess;
            int right_count = int(sum_right_hessian * cnt_factor + 0.5f);
            float sum_left_gradient = sum_gradients - sum_right_gradient;
            float sum_left_hessian  = sum_hessians - sum_right_hessian;
            int left_count = int(num_data) - right_count;

            if (sum_left_hessian >= min_sum_hessian_in_leaf &&
                left_count >= min_data_in_leaf &&
                sum_right_hessian >= min_sum_hessian_in_leaf &&
                right_count >= min_data_in_leaf) {
                float current_gain = GetSplitGain(
                    sum_left_gradient, sum_left_hessian,
                    sum_right_gradient, sum_right_hessian,
                    lambda_l1, lambda_l2);
                if (current_gain > min_gain_shift) {
                    local_best_gain = current_gain - min_gain_shift;
                    threshold_value = task.num_bin - 2 - tid_in_group;
                    threshold_found = true;
                    best_left_grad = sum_left_gradient;
                    best_left_hess = sum_left_hessian;
                }
            }
        }
    } else {
        // Forward: prefix_grad/prefix_hess is the left side
        uint end = (task.na_as_missing && task.mfb_offset == 1)
            ? (task.num_bin - 2)
            : (feature_num_bin_minus_offset - 2);
        bool skip = task.skip_default_bin &&
                    ((tid_in_group + task.mfb_offset) == task.default_bin);
        if (tid_in_group <= end && !skip) {
            float sum_left_gradient = prefix_grad;
            float sum_left_hessian  = prefix_hess;
            int left_count = int(sum_left_hessian * cnt_factor + 0.5f);
            float sum_right_gradient = sum_gradients - sum_left_gradient;
            float sum_right_hessian  = sum_hessians - sum_left_hessian;
            int right_count = int(num_data) - left_count;

            if (sum_left_hessian >= min_sum_hessian_in_leaf &&
                left_count >= min_data_in_leaf &&
                sum_right_hessian >= min_sum_hessian_in_leaf &&
                right_count >= min_data_in_leaf) {
                float current_gain = GetSplitGain(
                    sum_left_gradient, sum_left_hessian,
                    sum_right_gradient, sum_right_hessian,
                    lambda_l1, lambda_l2);
                if (current_gain > min_gain_shift) {
                    local_best_gain = current_gain - min_gain_shift;
                    threshold_value = (task.na_as_missing && task.mfb_offset == 1)
                        ? tid_in_group
                        : (tid_in_group + task.mfb_offset);
                    threshold_found = true;
                    best_left_grad = sum_left_gradient;
                    best_left_hess = sum_left_hessian;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4: Reduce best gain across the threadgroup
    // -----------------------------------------------------------------------
    float gain_for_reduce = threshold_found ? local_best_gain : kMinScore;
    uint  idx_for_reduce  = tid_in_group;

    uint num_simd_groups = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;
    simd_reduce_max_with_index(gain_for_reduce, idx_for_reduce,
                               shared_gain, shared_idx,
                               simd_lane_id, simd_group_id, num_simd_groups);
    // After reduction, shared_gain[0] = best gain, shared_idx[0] = best thread index

    uint best_thread = shared_idx[0];

    // -----------------------------------------------------------------------
    // Phase 5: Winning thread writes the result
    // -----------------------------------------------------------------------
    if (threshold_found && tid_in_group == best_thread) {
        out.is_valid = 1;
        out.gain = local_best_gain;
        out.threshold = threshold_value;
        out.inner_feature_index = task.inner_feature_index;
        out.default_left = uchar(task.assume_out_default_left);

        float final_left_grad, final_left_hess;
        float final_right_grad, final_right_hess;

        if (task.reverse) {
            float sum_right_grad = prefix_grad;
            float sum_right_hess = prefix_hess - kEpsilon;
            int right_count = int(sum_right_hess * cnt_factor + 0.5f);
            float sum_left_grad = sum_gradients - sum_right_grad;
            float sum_left_hess = sum_hessians - sum_right_hess - kEpsilon;
            int left_count = int(num_data) - right_count;

            final_left_grad  = sum_left_grad;
            final_left_hess  = sum_left_hess;
            final_right_grad = sum_right_grad;
            final_right_hess = sum_right_hess;

            out.left_sum_gradients  = sum_left_grad;
            out.left_sum_hessians   = sum_left_hess;
            out.left_count          = uint(left_count);
            out.right_sum_gradients = sum_right_grad;
            out.right_sum_hessians  = sum_right_hess;
        } else {
            float sum_left_grad = prefix_grad;
            float sum_left_hess = prefix_hess - kEpsilon;
            int left_count = int(sum_left_hess * cnt_factor + 0.5f);
            float sum_right_grad = sum_gradients - sum_left_grad;
            float sum_right_hess = sum_hessians - sum_left_hess - kEpsilon;
            // right_count = num_data - left_count (not stored separately)

            final_left_grad  = sum_left_grad;
            final_left_hess  = sum_left_hess;
            final_right_grad = sum_right_grad;
            final_right_hess = sum_right_hess;

            out.left_sum_gradients  = sum_left_grad;
            out.left_sum_hessians   = sum_left_hess;
            out.left_count          = uint(left_count);
            out.right_sum_gradients = sum_right_grad;
            out.right_sum_hessians  = sum_right_hess;
        }

        float left_output  = CalculateLeafOutput(final_left_grad, final_left_hess,
                                                  lambda_l1, lambda_l2);
        float right_output = CalculateLeafOutput(final_right_grad, final_right_hess,
                                                  lambda_l1, lambda_l2);
        out.left_value  = left_output;
        out.left_gain   = GetLeafGainGivenOutput(final_left_grad, final_left_hess,
                                                  lambda_l1, lambda_l2, left_output);
        out.right_value = right_output;
        out.right_gain  = GetLeafGainGivenOutput(final_right_grad, final_right_hess,
                                                  lambda_l1, lambda_l2, right_output);
    }
}


// ===========================================================================
// KERNEL 4b: find_best_split_global_memory
//
// Same as find_best_split but for features with more bins than can fit in
// registers (>1024 bins). Uses device memory buffers for prefix sums.
//
// Dispatch: threadgroups = num_tasks
//           threads_per_threadgroup = 256 or 1024
// ===========================================================================

kernel void find_best_split_global_memory(
    // Per-feature task descriptions
    const device SplitFindTask* tasks             [[buffer(0)]],
    const device uint&          num_tasks         [[buffer(1)]],
    // Feature-usage mask
    const device char*          is_feature_used   [[buffer(2)]],
    // Leaf information
    const device LeafSplitsStruct* leaf_splits    [[buffer(3)]],
    // Histogram buffer
    const device float*         histogram         [[buffer(4)]],
    // Regularization parameters
    const device float&         lambda_l1         [[buffer(5)]],
    const device float&         lambda_l2         [[buffer(6)]],
    const device float&         min_gain_to_split [[buffer(7)]],
    const device int&           min_data_in_leaf  [[buffer(8)]],
    const device float&         min_sum_hessian_in_leaf [[buffer(9)]],
    // Output: one SplitInfo per task
    device SplitInfo*           output_splits     [[buffer(10)]],
    // Global prefix sum buffers: each task gets max_bins_per_feature entries
    device float*               grad_prefix_buf   [[buffer(11)]],
    device float*               hess_prefix_buf   [[buffer(12)]],
    const device uint&          max_bins_per_feature [[buffer(13)]],
    // Threadgroup / thread identification
    uint group_id      [[threadgroup_position_in_grid]],
    uint tid_in_group  [[thread_position_in_threadgroup]],
    uint tg_size       [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (group_id >= num_tasks) return;

    const device SplitFindTask& task = tasks[group_id];
    device SplitInfo& out = output_splits[group_id];

    out.is_valid = 0;
    if (!is_feature_used[task.inner_feature_index]) return;

    const float sum_gradients = leaf_splits->sum_of_gradients;
    const float sum_hessians  = leaf_splits->sum_of_hessians + 2.0f * kEpsilon;
    const uint  num_data      = leaf_splits->num_data_in_leaf;
    const float parent_gain   = leaf_splits->gain;
    const float cnt_factor    = float(num_data) / sum_hessians;
    const float min_gain_shift = parent_gain + min_gain_to_split;

    const uint hist_base = task.hist_offset << 1;
    const device float* feature_hist_ptr = histogram + hist_base;
    const uint feature_num_bin_minus_offset = task.num_bin - task.mfb_offset;

    // Per-task prefix buffers
    device float* grad_buf = grad_prefix_buf + group_id * max_bins_per_feature;
    device float* hess_buf = hess_prefix_buf + group_id * max_bins_per_feature;

    // Threadgroup shared memory for reductions
    threadgroup float shared_gain[32];
    threadgroup uint  shared_idx[32];

    // -----------------------------------------------------------------------
    // Phase 1: Load histogram bins into device buffers
    // -----------------------------------------------------------------------
    if (!task.reverse) {
        if (task.na_as_missing && task.mfb_offset == 1) {
            // Gather non-default bins, then compute NA bin
            threadgroup float shared_prefix[32];
            float thread_sum_grad = 0.0f;
            float thread_sum_hess = 0.0f;
            for (uint bin = (tid_in_group > 0 ? tid_in_group : tg_size);
                 bin < task.num_bin; bin += tg_size) {
                uint bin_offset = (bin - 1) << 1;
                float g = feature_hist_ptr[bin_offset];
                float h = feature_hist_ptr[bin_offset + 1];
                grad_buf[bin] = g;
                hess_buf[bin] = h;
                thread_sum_grad += g;
                thread_sum_hess += h;
            }
            float total_g = threadgroup_reduce_sum(thread_sum_grad, shared_prefix,
                                                   tid_in_group, tg_size);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total_h = threadgroup_reduce_sum(thread_sum_hess, shared_prefix,
                                                   tid_in_group, tg_size);
            if (tid_in_group == 0) {
                grad_buf[0] = sum_gradients - total_g;
                hess_buf[0] = sum_hessians - total_h;
            }
        } else {
            for (uint bin = tid_in_group; bin < feature_num_bin_minus_offset; bin += tg_size) {
                bool skip = task.skip_default_bin &&
                            ((bin + task.mfb_offset) == task.default_bin);
                if (!skip) {
                    uint bin_offset = bin << 1;
                    grad_buf[bin] = feature_hist_ptr[bin_offset];
                    hess_buf[bin] = feature_hist_ptr[bin_offset + 1];
                } else {
                    grad_buf[bin] = 0.0f;
                    hess_buf[bin] = 0.0f;
                }
            }
        }
    } else {
        for (uint bin = tid_in_group; bin < feature_num_bin_minus_offset; bin += tg_size) {
            bool skip = (bin >= uint(task.na_as_missing)) &&
                        task.skip_default_bin &&
                        ((task.num_bin - 1 - bin) == task.default_bin);
            if (!skip) {
                uint read_index = feature_num_bin_minus_offset - 1 - bin;
                uint bin_offset = read_index << 1;
                grad_buf[bin] = feature_hist_ptr[bin_offset];
                hess_buf[bin] = feature_hist_ptr[bin_offset + 1];
            } else {
                grad_buf[bin] = 0.0f;
                hess_buf[bin] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid_in_group == 0) {
        hess_buf[0] += kEpsilon;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // -----------------------------------------------------------------------
    // Phase 2: Sequential prefix sum in device memory (Blelloch-style would
    // be better for very large bin counts, but sequential is correct and
    // the bin count rarely exceeds a few thousand).
    // -----------------------------------------------------------------------
    // Use a simple parallel Hillis-Steele-like approach with multiple passes
    if (tid_in_group == 0) {
        for (uint i = 1; i < feature_num_bin_minus_offset; ++i) {
            grad_buf[i] += grad_buf[i - 1];
            hess_buf[i] += hess_buf[i - 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // -----------------------------------------------------------------------
    // Phase 3: Each thread evaluates gain for its range of bins
    // -----------------------------------------------------------------------
    float local_best_gain = kMinScore;
    bool  threshold_found = false;
    uint  threshold_value = 0;
    float best_prefix_grad = 0.0f;
    float best_prefix_hess = 0.0f;

    if (task.reverse) {
        for (uint bin = tid_in_group; bin < feature_num_bin_minus_offset; bin += tg_size) {
            bool skip = (bin >= uint(task.na_as_missing)) &&
                        task.skip_default_bin &&
                        ((task.num_bin - 1 - bin) == task.default_bin);
            if (!skip && bin >= uint(task.na_as_missing) && bin <= task.num_bin - 2) {
                float sum_right_gradient = grad_buf[bin];
                float sum_right_hessian  = hess_buf[bin];
                int right_count = int(sum_right_hessian * cnt_factor + 0.5f);
                float sum_left_gradient = sum_gradients - sum_right_gradient;
                float sum_left_hessian  = sum_hessians - sum_right_hessian;
                int left_count = int(num_data) - right_count;

                if (sum_left_hessian >= min_sum_hessian_in_leaf &&
                    left_count >= min_data_in_leaf &&
                    sum_right_hessian >= min_sum_hessian_in_leaf &&
                    right_count >= min_data_in_leaf) {
                    float current_gain = GetSplitGain(
                        sum_left_gradient, sum_left_hessian,
                        sum_right_gradient, sum_right_hessian,
                        lambda_l1, lambda_l2);
                    if (current_gain > min_gain_shift && (current_gain - min_gain_shift) > local_best_gain) {
                        local_best_gain = current_gain - min_gain_shift;
                        threshold_value = task.num_bin - 2 - bin;
                        threshold_found = true;
                        best_prefix_grad = sum_right_gradient;
                        best_prefix_hess = sum_right_hessian;
                    }
                }
            }
        }
    } else {
        uint end = (task.na_as_missing && task.mfb_offset == 1)
            ? (task.num_bin - 2) : (feature_num_bin_minus_offset - 2);
        for (uint bin = tid_in_group; bin <= end; bin += tg_size) {
            bool skip = task.skip_default_bin &&
                        ((bin + task.mfb_offset) == task.default_bin);
            if (!skip) {
                float sum_left_gradient = grad_buf[bin];
                float sum_left_hessian  = hess_buf[bin];
                int left_count = int(sum_left_hessian * cnt_factor + 0.5f);
                float sum_right_gradient = sum_gradients - sum_left_gradient;
                float sum_right_hessian  = sum_hessians - sum_left_hessian;
                int right_count = int(num_data) - left_count;

                if (sum_left_hessian >= min_sum_hessian_in_leaf &&
                    left_count >= min_data_in_leaf &&
                    sum_right_hessian >= min_sum_hessian_in_leaf &&
                    right_count >= min_data_in_leaf) {
                    float current_gain = GetSplitGain(
                        sum_left_gradient, sum_left_hessian,
                        sum_right_gradient, sum_right_hessian,
                        lambda_l1, lambda_l2);
                    if (current_gain > min_gain_shift && (current_gain - min_gain_shift) > local_best_gain) {
                        local_best_gain = current_gain - min_gain_shift;
                        threshold_value = (task.na_as_missing && task.mfb_offset == 1)
                            ? bin : (bin + task.mfb_offset);
                        threshold_found = true;
                        best_prefix_grad = sum_left_gradient;
                        best_prefix_hess = sum_left_hessian;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4: Reduce best gain across threadgroup
    // -----------------------------------------------------------------------
    float gain_for_reduce = threshold_found ? local_best_gain : kMinScore;
    uint  idx_for_reduce  = tid_in_group;
    uint num_simd_groups = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;
    simd_reduce_max_with_index(gain_for_reduce, idx_for_reduce,
                               shared_gain, shared_idx,
                               simd_lane_id, simd_group_id, num_simd_groups);
    uint best_thread = shared_idx[0];

    // -----------------------------------------------------------------------
    // Phase 5: Winner writes result
    // -----------------------------------------------------------------------
    if (threshold_found && tid_in_group == best_thread) {
        out.is_valid = 1;
        out.gain = local_best_gain;
        out.threshold = threshold_value;
        out.inner_feature_index = task.inner_feature_index;
        out.default_left = uchar(task.assume_out_default_left);

        float final_left_grad, final_left_hess;
        float final_right_grad, final_right_hess;
        int left_count_final, right_count_final;

        if (task.reverse) {
            uint best_bin = task.num_bin - 2 - threshold_value;
            float sum_right_grad = grad_buf[best_bin];
            float sum_right_hess = hess_buf[best_bin] - kEpsilon;
            right_count_final = int(sum_right_hess * cnt_factor + 0.5f);
            float sum_left_grad = sum_gradients - sum_right_grad;
            float sum_left_hess = sum_hessians - sum_right_hess - kEpsilon;
            left_count_final = int(num_data) - right_count_final;

            final_left_grad  = sum_left_grad;
            final_left_hess  = sum_left_hess;
            final_right_grad = sum_right_grad;
            final_right_hess = sum_right_hess;
        } else {
            uint best_bin = (task.na_as_missing && task.mfb_offset == 1)
                ? threshold_value : (threshold_value - task.mfb_offset);
            float sum_left_grad = grad_buf[best_bin];
            float sum_left_hess = hess_buf[best_bin] - kEpsilon;
            left_count_final = int(sum_left_hess * cnt_factor + 0.5f);
            float sum_right_grad = sum_gradients - sum_left_grad;
            float sum_right_hess = sum_hessians - sum_left_hess - kEpsilon;
            right_count_final = int(num_data) - left_count_final;

            final_left_grad  = sum_left_grad;
            final_left_hess  = sum_left_hess;
            final_right_grad = sum_right_grad;
            final_right_hess = sum_right_hess;
        }

        out.left_sum_gradients  = final_left_grad;
        out.left_sum_hessians   = final_left_hess;
        out.left_count          = uint(max(left_count_final, 0));
        out.right_sum_gradients = final_right_grad;
        out.right_sum_hessians  = final_right_hess;

        float left_output  = CalculateLeafOutput(final_left_grad, final_left_hess,
                                                  lambda_l1, lambda_l2);
        float right_output = CalculateLeafOutput(final_right_grad, final_right_hess,
                                                  lambda_l1, lambda_l2);
        out.left_value  = left_output;
        out.left_gain   = GetLeafGainGivenOutput(final_left_grad, final_left_hess,
                                                  lambda_l1, lambda_l2, left_output);
        out.right_value = right_output;
        out.right_gain  = GetLeafGainGivenOutput(final_right_grad, final_right_hess,
                                                  lambda_l1, lambda_l2, right_output);
    }
}


// ===========================================================================
// KERNEL 5: sync_best_split
//
// Reduce per-feature best splits to find the single best split per leaf.
// Analogous to SyncBestSplitForLeafKernel in CUDA.
//
// Input:  per_feature_splits[0..num_tasks-1]  (smaller leaf)
//         per_feature_splits[num_tasks..2*num_tasks-1] (larger leaf)
// Output: leaf_best_splits[leaf_index] for each leaf
//
// Dispatch: threadgroups = num_blocks_per_leaf * (1 or 2 for smaller/larger)
//           threads_per_threadgroup = 256 (NUM_TASKS_PER_SYNC_BLOCK)
// ===========================================================================

kernel void sync_best_split(
    const device SplitInfo*    per_feature_splits [[buffer(0)]],
    const device SplitFindTask* tasks             [[buffer(1)]],
    device SplitInfo*          leaf_best_splits   [[buffer(2)]],
    const device int&          num_tasks          [[buffer(3)]],
    const device int&          smaller_leaf_index [[buffer(4)]],
    const device int&          larger_leaf_index  [[buffer(5)]],
    const device int&          num_leaves         [[buffer(6)]],
    const device int&          num_blocks_per_leaf [[buffer(7)]],
    const device int&          is_larger_only     [[buffer(8)]],
    uint group_id      [[threadgroup_position_in_grid]],
    uint tid_in_group  [[thread_position_in_threadgroup]],
    uint tg_size       [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float shared_gain[32];
    threadgroup uint  shared_idx[32];

    bool is_smaller = (int(group_id) < num_blocks_per_leaf) && (is_larger_only == 0);
    uint leaf_block_index = (is_smaller || (is_larger_only != 0))
        ? group_id
        : (group_id - uint(num_blocks_per_leaf));

    int task_index = int(leaf_block_index * tg_size + tid_in_group);
    uint read_index = is_smaller
        ? uint(task_index)
        : uint(task_index + num_tasks);

    bool best_found = false;
    float best_gain = kMinScore;
    uint best_read_idx = read_index;

    if (task_index < num_tasks) {
        best_found = (per_feature_splits[read_index].is_valid != 0);
        best_gain  = per_feature_splits[read_index].gain;
        best_read_idx = read_index;
    }

    // Reduce: find max gain and its read_index across threadgroup
    float gain_for_reduce = best_found ? best_gain : kMinScore;
    uint  idx_for_reduce  = best_read_idx;
    uint num_simd_groups = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;
    simd_reduce_max_with_index(gain_for_reduce, idx_for_reduce,
                               shared_gain, shared_idx,
                               simd_lane_id, simd_group_id, num_simd_groups);

    if (tid_in_group == 0) {
        uint best_idx = shared_idx[0];
        int leaf_index_ref = is_smaller ? smaller_leaf_index : larger_leaf_index;
        uint buffer_write_pos = uint(leaf_index_ref) + leaf_block_index * uint(num_leaves);
        device SplitInfo& dst = leaf_best_splits[buffer_write_pos];

        if (per_feature_splits[best_idx].is_valid != 0) {
            dst = per_feature_splits[best_idx];
            // Fix up inner_feature_index from the task
            int task_for_best = is_smaller
                ? int(best_idx)
                : (int(best_idx) - num_tasks);
            dst.inner_feature_index = tasks[task_for_best].inner_feature_index;
            dst.is_valid = 1;
        } else {
            dst.gain = kMinScore;
            dst.is_valid = 0;
        }
    }
}


// ===========================================================================
// KERNEL 5b: sync_best_split_all_blocks
//
// When multiple blocks were used per leaf in sync_best_split, this kernel
// reduces across blocks to get the single best split per leaf.
// Analogous to SyncBestSplitForLeafKernelAllBlocks in CUDA.
//
// Dispatch: threadgroups = 2 (one for smaller, one for larger)
//           threads_per_threadgroup = 1
// ===========================================================================

kernel void sync_best_split_all_blocks(
    device SplitInfo*     leaf_best_splits     [[buffer(0)]],
    const device int&     smaller_leaf_index   [[buffer(1)]],
    const device int&     larger_leaf_index    [[buffer(2)]],
    const device uint&    num_blocks_per_leaf  [[buffer(3)]],
    const device int&     num_leaves           [[buffer(4)]],
    const device int&     is_larger_only       [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]])
{
    // Block 0 handles smaller leaf (unless larger_only)
    // Block 1 handles larger leaf
    if (group_id == 0 && (is_larger_only == 0)) {
        device SplitInfo& best = leaf_best_splits[smaller_leaf_index];
        for (uint block_idx = 1; block_idx < num_blocks_per_leaf; ++block_idx) {
            uint read_pos = uint(smaller_leaf_index) + block_idx * uint(num_leaves);
            const device SplitInfo& other = leaf_best_splits[read_pos];
            if ((other.is_valid != 0 && best.is_valid != 0 && other.gain > best.gain) ||
                (best.is_valid == 0 && other.is_valid != 0)) {
                best = other;
            }
        }
    }
    if (larger_leaf_index >= 0) {
        if (group_id == 1 || (is_larger_only != 0)) {
            device SplitInfo& best = leaf_best_splits[larger_leaf_index];
            for (uint block_idx = 1; block_idx < num_blocks_per_leaf; ++block_idx) {
                uint read_pos = uint(larger_leaf_index) + block_idx * uint(num_leaves);
                const device SplitInfo& other = leaf_best_splits[read_pos];
                if ((other.is_valid != 0 && best.is_valid != 0 && other.gain > best.gain) ||
                    (best.is_valid == 0 && other.is_valid != 0)) {
                    best = other;
                }
            }
        }
    }
}


// ===========================================================================
// KERNEL 6: find_best_from_all
//
// Find the globally best leaf to split across all current leaves.
// Analogous to FindBestFromAllSplitsKernel in CUDA.
//
// Dispatch: threadgroups = 1
//           threads_per_threadgroup = max(32, cur_num_leaves rounded up to 32)
// ===========================================================================

kernel void find_best_from_all(
    device SplitInfo*          leaf_best_splits   [[buffer(0)]],
    const device int&          cur_num_leaves     [[buffer(1)]],
    // Output: index of the best leaf to split
    device int*                best_leaf_out      [[buffer(2)]],
    // Thread identification
    uint tid_in_group  [[thread_position_in_threadgroup]],
    uint tg_size       [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float shared_gain[32];
    threadgroup int   shared_leaf[32];

    float thread_best_gain = kMinScore;
    int   thread_best_leaf = -1;

    // Each thread examines a subset of leaves
    for (int leaf_index = int(tid_in_group);
         leaf_index < cur_num_leaves;
         leaf_index += int(tg_size)) {
        if (leaf_best_splits[leaf_index].is_valid != 0) {
            float g = leaf_best_splits[leaf_index].gain;
            if (g > thread_best_gain) {
                thread_best_gain = g;
                thread_best_leaf = leaf_index;
            }
        }
    }

    // Intra-SIMD reduction
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset >>= 1) {
        float other_gain = simd_shuffle_down(thread_best_gain, offset);
        int   other_leaf = simd_shuffle_down(thread_best_leaf, offset);
        if ((thread_best_leaf != -1 && other_leaf != -1 && other_gain > thread_best_gain) ||
            (thread_best_leaf == -1 && other_leaf != -1)) {
            thread_best_gain = other_gain;
            thread_best_leaf = other_leaf;
        }
    }

    uint simd_lane = tid_in_group % SIMD_SIZE;
    uint sg_id     = tid_in_group / SIMD_SIZE;
    uint num_simds = (tg_size + SIMD_SIZE - 1) / SIMD_SIZE;

    if (simd_lane == 0) {
        shared_gain[sg_id] = thread_best_gain;
        shared_leaf[sg_id] = thread_best_leaf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0) {
        float g = (simd_lane < num_simds) ? shared_gain[simd_lane] : kMinScore;
        int   l = (simd_lane < num_simds) ? shared_leaf[simd_lane] : -1;
        for (uint offset = SIMD_SIZE / 2; offset > 0; offset >>= 1) {
            float other_g = simd_shuffle_down(g, offset);
            int   other_l = simd_shuffle_down(l, offset);
            if ((l != -1 && other_l != -1 && other_g > g) ||
                (l == -1 && other_l != -1)) {
                g = other_g;
                l = other_l;
            }
        }
        if (simd_lane == 0) {
            best_leaf_out[0] = l;
            // Mark the chosen leaf as consumed so it isn't picked again
            if (l >= 0) {
                leaf_best_splits[l].is_valid = 0;
            }
        }
    }
}


// ===========================================================================
// KERNEL 7: init_leaf_values
//
// Parallel reduction to compute the sum of gradients and hessians for the
// root leaf (or any initial leaf). Two-pass approach matching CUDA:
//   Pass 1 (this kernel): each threadgroup sums a block of data, writes
//           per-block partial sums to a buffer.
//   Pass 2 (init_leaf_values_reduce): single threadgroup reduces the
//           partial sums and writes the LeafSplitsStruct.
//
// Dispatch: threadgroups = ceil(num_data / threads_per_threadgroup)
//           threads_per_threadgroup = 256
// ===========================================================================

kernel void init_leaf_values(
    const device float*   gradients             [[buffer(0)]],
    const device float*   hessians              [[buffer(1)]],
    const device int*     data_indices          [[buffer(2)]],
    const device uint&    num_data              [[buffer(3)]],
    const device int&     use_indices           [[buffer(4)]],
    device float*         block_sum_gradients   [[buffer(5)]],
    device float*         block_sum_hessians    [[buffer(6)]],
    uint group_id      [[threadgroup_position_in_grid]],
    uint tid_in_group  [[thread_position_in_threadgroup]],
    uint tg_size       [[threads_per_threadgroup]])
{
    uint global_tid = group_id * tg_size + tid_in_group;

    float grad_val = 0.0f;
    float hess_val = 0.0f;
    if (global_tid < num_data) {
        int data_index = (use_indices != 0) ? data_indices[global_tid] : int(global_tid);
        grad_val = gradients[data_index];
        hess_val = hessians[data_index];
    }

    // Threadgroup reduction
    threadgroup float shared_buf[32];
    float sum_grad = threadgroup_reduce_sum(grad_val, shared_buf, tid_in_group, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_hess = threadgroup_reduce_sum(hess_val, shared_buf, tid_in_group, tg_size);

    if (tid_in_group == 0) {
        block_sum_gradients[group_id] = sum_grad;
        block_sum_hessians[group_id]  = sum_hess;
    }
}


// ===========================================================================
// KERNEL 7b: init_leaf_values_reduce
//
// Second pass: reduce per-block partial sums from init_leaf_values,
// compute the root leaf gain and output value, and write the
// LeafSplitsStruct.
//
// Dispatch: threadgroups = 1
//           threads_per_threadgroup = 256
// ===========================================================================

kernel void init_leaf_values_reduce(
    device float*              block_sum_gradients [[buffer(0)]],
    device float*              block_sum_hessians  [[buffer(1)]],
    const device uint&         num_blocks          [[buffer(2)]],
    const device uint&         num_data            [[buffer(3)]],
    const device float&        lambda_l1           [[buffer(4)]],
    const device float&        lambda_l2           [[buffer(5)]],
    const device uint&         data_indices_offset [[buffer(6)]],
    const device uint&         hist_offset         [[buffer(7)]],
    device LeafSplitsStruct*   leaf_struct         [[buffer(8)]],
    uint tid_in_group  [[thread_position_in_threadgroup]],
    uint tg_size       [[threads_per_threadgroup]])
{
    // Each thread accumulates a portion of the block sums
    float thread_sum_grad = 0.0f;
    float thread_sum_hess = 0.0f;
    for (uint i = tid_in_group; i < num_blocks; i += tg_size) {
        thread_sum_grad += block_sum_gradients[i];
        thread_sum_hess += block_sum_hessians[i];
    }

    threadgroup float shared_buf[32];
    float sum_grad = threadgroup_reduce_sum(thread_sum_grad, shared_buf, tid_in_group, tg_size);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_hess = threadgroup_reduce_sum(thread_sum_hess, shared_buf, tid_in_group, tg_size);

    if (tid_in_group == 0) {
        leaf_struct->leaf_index       = 0;
        leaf_struct->sum_of_gradients = sum_grad;
        leaf_struct->sum_of_hessians  = sum_hess;
        leaf_struct->num_data_in_leaf = num_data;
        leaf_struct->gain             = GetLeafGain(sum_grad, sum_hess, lambda_l1, lambda_l2);
        leaf_struct->leaf_value       = CalculateLeafOutput(sum_grad, sum_hess, lambda_l1, lambda_l2);
        leaf_struct->data_indices_offset = data_indices_offset;
        leaf_struct->hist_offset      = hist_offset;

        // Also write back reduced values for host readback if needed
        block_sum_gradients[0] = sum_grad;
        block_sum_hessians[0]  = sum_hess;
    }
}


// ===========================================================================
// KERNEL 7c: init_leaf_values_empty
//
// Initialize an empty leaf (e.g., larger_leaf at root when there's no split).
//
// Dispatch: threadgroups = 1, threads_per_threadgroup = 1
// ===========================================================================

kernel void init_leaf_values_empty(
    device LeafSplitsStruct* leaf_struct [[buffer(0)]])
{
    leaf_struct->leaf_index       = -1;
    leaf_struct->sum_of_gradients = 0.0f;
    leaf_struct->sum_of_hessians  = 0.0f;
    leaf_struct->num_data_in_leaf = 0;
    leaf_struct->gain             = 0.0f;
    leaf_struct->leaf_value       = 0.0f;
    leaf_struct->data_indices_offset = 0;
    leaf_struct->hist_offset      = 0;
}


// ===========================================================================
// KERNEL 8: prepare_leaf_best_split_info
//
// Extract key fields from the per-leaf best split info into a flat int
// buffer for easy host readback. Matches PrepareLeafBestSplitInfo in CUDA.
//
// Dispatch: threadgroups = 6 (or 3 if no larger leaf)
//           threads_per_threadgroup = 1
// ===========================================================================

kernel void prepare_leaf_best_split_info(
    const device SplitInfo*    leaf_best_splits     [[buffer(0)]],
    device int*                best_split_buffer    [[buffer(1)]],
    const device int&          smaller_leaf_index   [[buffer(2)]],
    const device int&          larger_leaf_index    [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]])
{
    if (group_id == 0) {
        best_split_buffer[0] = leaf_best_splits[smaller_leaf_index].inner_feature_index;
    } else if (group_id == 1) {
        best_split_buffer[1] = int(leaf_best_splits[smaller_leaf_index].threshold);
    } else if (group_id == 2) {
        best_split_buffer[2] = int(leaf_best_splits[smaller_leaf_index].default_left);
    }
    if (larger_leaf_index >= 0) {
        if (group_id == 3) {
            best_split_buffer[3] = leaf_best_splits[larger_leaf_index].inner_feature_index;
        } else if (group_id == 4) {
            best_split_buffer[4] = int(leaf_best_splits[larger_leaf_index].threshold);
        } else if (group_id == 5) {
            best_split_buffer[5] = int(leaf_best_splits[larger_leaf_index].default_left);
        }
    }
}


// ===========================================================================
// KERNEL 9: set_invalid_leaf_split_info
//
// Mark a leaf's best split as invalid when the leaf is not valid for
// splitting (e.g., below min_data_in_leaf or not selected).
//
// Dispatch: threadgroups = 1, threads_per_threadgroup = 1
// ===========================================================================

kernel void set_invalid_leaf_split_info(
    device SplitInfo*     leaf_best_splits       [[buffer(0)]],
    const device int&     is_smaller_leaf_valid  [[buffer(1)]],
    const device int&     is_larger_leaf_valid   [[buffer(2)]],
    const device int&     smaller_leaf_index     [[buffer(3)]],
    const device int&     larger_leaf_index      [[buffer(4)]])
{
    if (is_smaller_leaf_valid == 0) {
        leaf_best_splits[smaller_leaf_index].is_valid = 0;
    }
    if (is_larger_leaf_valid == 0 && larger_leaf_index >= 0) {
        leaf_best_splits[larger_leaf_index].is_valid = 0;
    }
}
