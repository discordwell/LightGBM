/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * Metal GPU kernels for LightGBM histogram construction.
 * Compiled to lib_lightgbm.metallib at build time via xcrun -sdk macosx metallib.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

inline void atomic_add_float_tg(threadgroup atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    do {
        next = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, next,
                memory_order_relaxed, memory_order_relaxed));
}

// ===========================================================================
// gather_to_leaf_order — reorder grad/hess/bins into leaf-sequential order
//
// Eliminates ALL random-access indirection from the histogram kernel.
// One thread per leaf row. Copies:
//   - gradients/hessians from scattered positions to sequential
//   - bin data from column-major [groups × total_rows] scattered
//     to column-major [groups × leaf_rows] sequential
//
// After this kernel, the histogram kernel reads everything sequentially.
// This is the same optimization the CPU does with ordered_gradients.
// ===========================================================================

kernel void gather_to_leaf_order(
    const device float*  src_grad       [[buffer(0)]],
    const device float*  src_hess       [[buffer(1)]],
    const device int*    data_indices   [[buffer(2)]],
    device float*        dst_grad       [[buffer(3)]],
    device float*        dst_hess       [[buffer(4)]],
    const device uchar*  src_bins       [[buffer(5)]],  // col-major [groups × total_rows]
    device uchar*        dst_bins       [[buffer(6)]],  // col-major [groups × leaf_rows]
    constant uint&       num_data       [[buffer(7)]],
    constant uint&       num_data_total [[buffer(8)]],
    constant uint&       num_groups     [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_data) return;
    const int row = data_indices[gid];
    dst_grad[gid] = src_grad[row];
    dst_hess[gid] = src_hess[row];

    // Gather bin values: column-major src → column-major dst
    for (uint g = 0; g < num_groups; g++) {
        dst_bins[(uint64_t)g * num_data + gid] =
            src_bins[(uint64_t)g * num_data_total + row];
    }
}

// ===========================================================================
// gather_packed_to_leaf_order — reorder grad / hess and 4-feature tuples
// ===========================================================================

kernel void gather_packed_to_leaf_order(
    const device float*   src_grad       [[buffer(0)]],
    const device float*   src_hess       [[buffer(1)]],
    const device int*     data_indices   [[buffer(2)]],
    device float*         dst_grad       [[buffer(3)]],
    device float*         dst_hess       [[buffer(4)]],
    const device uchar4*  src_bins       [[buffer(5)]],  // [tuples × total_rows]
    device uchar4*        dst_bins       [[buffer(6)]],  // [tuples × leaf_rows]
    constant uint&        num_data       [[buffer(7)]],
    constant uint&        num_data_total [[buffer(8)]],
    constant uint&        num_tuples     [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_data) return;
    const int row = data_indices[gid];
    dst_grad[gid] = src_grad[row];
    dst_hess[gid] = src_hess[row];

    for (uint tuple = 0; tuple < num_tuples; ++tuple) {
        dst_bins[(uint64_t)tuple * num_data + gid] =
            src_bins[(uint64_t)tuple * num_data_total + row];
    }
}

// ===========================================================================
// histogram_gathered — column-grouped with pre-gathered sequential data
//
// One threadgroup per feature group. All reads are sequential:
//   - ordered_grad[i] — sequential, fits in GPU L2 cache across groups
//   - ordered_hess[i] — sequential, fits in GPU L2 cache
//   - gathered_bins[grp * N + i] — sequential per group
//
// Accumulates into threadgroup-local histogram with CAS-loop atomics.
// Threadgroup CAS is ~25x faster than device-memory CAS.
// ===========================================================================

kernel void histogram_gathered(
    const device float*   ordered_grad     [[buffer(0)]],
    const device float*   ordered_hess     [[buffer(1)]],
    const device uchar*   gathered_bins    [[buffer(2)]],  // col-major [groups × leaf_rows]
    const device uint*    group_bin_offsets [[buffer(3)]],
    device float*         hist_output      [[buffer(4)]],
    constant uint&        num_data_in_leaf [[buffer(5)]],
    constant uint&        num_groups       [[buffer(6)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    const uint grp = gid;
    if (grp >= num_groups) return;

    const uint bin_start = group_bin_offsets[grp];
    const uint nbins = group_bin_offsets[grp + 1] - bin_start;

    threadgroup atomic_uint local_hist[256 * 2];

    for (uint i = tid; i < nbins * 2; i += tgs) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All three reads are sequential — no data_indices indirection
    const device uchar* grp_bins = gathered_bins + (uint64_t)grp * num_data_in_leaf;

    for (uint i = tid; i < num_data_in_leaf; i += tgs) {
        const float g = ordered_grad[i];    // sequential, cached in L2 after first group
        const float h = ordered_hess[i];    // sequential, cached in L2
        const uint bin = grp_bins[i];       // sequential within this group

        atomic_add_float_tg(&local_hist[bin * 2], g);
        atomic_add_float_tg(&local_hist[bin * 2 + 1], h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < nbins * 2; i += tgs) {
        hist_output[(bin_start + i / 2) * 2 + (i % 2)] =
            as_type<float>(atomic_load_explicit(&local_hist[i], memory_order_relaxed));
    }
}

// ===========================================================================
// histogram_gathered_subhist — several sub-histograms per feature group
//
// Each workgroup owns a row partition for one feature group, builds a
// threadgroup-local histogram, and writes it to a scratch buffer.
// A later kernel reduces the sub-histograms into the final output.
//
// This is the key idea missing from the earlier Metal experiments:
// we don't need a single shared histogram per feature group.
// ===========================================================================

kernel void histogram_gathered_subhist(
    const device float*   ordered_grad      [[buffer(0)]],
    const device float*   ordered_hess      [[buffer(1)]],
    const device uchar*   gathered_bins     [[buffer(2)]],  // col-major [groups × leaf_rows]
    const device uint*    group_bin_offsets [[buffer(3)]],
    device float*         subhist_output    [[buffer(4)]],  // [groups * parts * 256 * 2]
    constant uint&        num_data_in_leaf  [[buffer(5)]],
    constant uint&        num_groups        [[buffer(6)]],
    constant uint&        num_parts         [[buffer(7)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    const uint grp = gid / num_parts;
    const uint part = gid % num_parts;
    if (grp >= num_groups) return;

    const uint nbins = group_bin_offsets[grp + 1] - group_bin_offsets[grp];
    threadgroup atomic_uint local_hist[256 * 2];

    for (uint i = tid; i < nbins * 2; i += tgs) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint start = static_cast<uint>((static_cast<uint64_t>(part) * num_data_in_leaf) / num_parts);
    const uint end = static_cast<uint>((static_cast<uint64_t>(part + 1) * num_data_in_leaf) / num_parts);
    const device uchar* grp_bins = gathered_bins + (uint64_t)grp * num_data_in_leaf;

    for (uint i = start + tid; i < end; i += tgs) {
        const float g = ordered_grad[i];
        const float h = ordered_hess[i];
        const uint bin = grp_bins[i];

        atomic_add_float_tg(&local_hist[bin * 2], g);
        atomic_add_float_tg(&local_hist[bin * 2 + 1], h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out = subhist_output + (uint64_t)gid * 256 * 2;
    for (uint i = tid; i < nbins * 2; i += tgs) {
        out[i] = as_type<float>(atomic_load_explicit(&local_hist[i], memory_order_relaxed));
    }
}

// ===========================================================================
// reduce_histogram_subhist — merge row partitions back into final histograms
// ===========================================================================

kernel void reduce_histogram_subhist(
    const device float*   subhist_input     [[buffer(0)]],  // [groups * parts * 256 * 2]
    const device uint*    group_bin_offsets [[buffer(1)]],
    device float*         hist_output       [[buffer(2)]],
    constant uint&        num_groups        [[buffer(3)]],
    constant uint&        num_parts         [[buffer(4)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    const uint grp = gid;
    if (grp >= num_groups) return;

    const uint bin_start = group_bin_offsets[grp];
    const uint nbins = group_bin_offsets[grp + 1] - bin_start;
    const device float* grp_subhist = subhist_input + (uint64_t)grp * num_parts * 256 * 2;

    for (uint i = tid; i < nbins * 2; i += tgs) {
        float sum = 0.0f;
        for (uint part = 0; part < num_parts; ++part) {
            sum += grp_subhist[(uint64_t)part * 256 * 2 + i];
        }
        hist_output[(bin_start + i / 2) * 2 + (i % 2)] = sum;
    }
}

// ===========================================================================
// histogram_packed_subhist — process 4 dense feature groups per workgroup
// ===========================================================================

kernel void histogram_packed_subhist(
    const device float*   ordered_grad      [[buffer(0)]],
    const device float*   ordered_hess      [[buffer(1)]],
    const device uchar4*  gathered_bins     [[buffer(2)]],  // [tuples × leaf_rows]
    const device uint*    dense_group_map   [[buffer(3)]],  // [tuples × 4]
    const device uint*    group_bin_offsets [[buffer(4)]],
    device float*         subhist_output    [[buffer(5)]],  // [tuples * parts * 4 * 256 * 2]
    constant uint&        num_data_in_leaf  [[buffer(6)]],
    constant uint&        num_tuples        [[buffer(7)]],
    constant uint&        num_parts         [[buffer(8)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    const uint tuple = gid / num_parts;
    const uint part = gid % num_parts;
    if (tuple >= num_tuples) return;

    threadgroup atomic_uint local_hist[4 * 256 * 2];
    for (uint i = tid; i < 4 * 256 * 2; i += tgs) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint start = (uint)((uint64_t)part * num_data_in_leaf / num_parts);
    const uint end = (uint)((uint64_t)(part + 1) * num_data_in_leaf / num_parts);
    const device uchar4* tuple_bins = gathered_bins + (uint64_t)tuple * num_data_in_leaf;
    const device uint* groups = dense_group_map + (uint64_t)tuple * 4;

    for (uint i = start + tid; i < end; i += tgs) {
        const float g = ordered_grad[i];
        const float h = ordered_hess[i];
        const uchar4 bins = tuple_bins[i];

        if (groups[0] != 0xFFFFFFFFu) {
            const uint bin = bins[0];
            atomic_add_float_tg(&local_hist[bin * 2], g);
            atomic_add_float_tg(&local_hist[bin * 2 + 1], h);
        }
        if (groups[1] != 0xFFFFFFFFu) {
            const uint bin = bins[1];
            atomic_add_float_tg(&local_hist[(256 + bin) * 2], g);
            atomic_add_float_tg(&local_hist[(256 + bin) * 2 + 1], h);
        }
        if (groups[2] != 0xFFFFFFFFu) {
            const uint bin = bins[2];
            atomic_add_float_tg(&local_hist[(512 + bin) * 2], g);
            atomic_add_float_tg(&local_hist[(512 + bin) * 2 + 1], h);
        }
        if (groups[3] != 0xFFFFFFFFu) {
            const uint bin = bins[3];
            atomic_add_float_tg(&local_hist[(768 + bin) * 2], g);
            atomic_add_float_tg(&local_hist[(768 + bin) * 2 + 1], h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out = subhist_output + (uint64_t)gid * 4 * 256 * 2;
    for (uint lane = 0; lane < 4; ++lane) {
        if (groups[lane] == 0xFFFFFFFFu) {
            continue;
        }
        const uint nbins = group_bin_offsets[groups[lane] + 1] - group_bin_offsets[groups[lane]];
        const uint lane_offset = lane * 256 * 2;
        for (uint i = tid; i < nbins * 2; i += tgs) {
            out[lane_offset + i] =
                as_type<float>(atomic_load_explicit(&local_hist[lane_offset + i], memory_order_relaxed));
        }
    }
}

// ===========================================================================
// reduce_histogram_packed_subhist — merge tuple partitions into final histograms
// ===========================================================================

kernel void reduce_histogram_packed_subhist(
    const device float*   subhist_input     [[buffer(0)]],  // [tuples * parts * 4 * 256 * 2]
    const device uint*    dense_group_map   [[buffer(1)]],  // [tuples × 4]
    const device uint*    group_bin_offsets [[buffer(2)]],
    device float*         hist_output       [[buffer(3)]],
    constant uint&        num_tuples        [[buffer(4)]],
    constant uint&        num_parts         [[buffer(5)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    const uint tuple = gid;
    if (tuple >= num_tuples) return;

    const device uint* groups = dense_group_map + (uint64_t)tuple * 4;
    const device float* tuple_subhist = subhist_input + (uint64_t)tuple * num_parts * 4 * 256 * 2;

    for (uint lane = 0; lane < 4; ++lane) {
        const uint group = groups[lane];
        if (group == 0xFFFFFFFFu) {
            continue;
        }
        const uint bin_start = group_bin_offsets[group];
        const uint nbins = group_bin_offsets[group + 1] - bin_start;
        const uint lane_offset = lane * 256 * 2;

        for (uint i = tid; i < nbins * 2; i += tgs) {
            float sum = 0.0f;
            for (uint part = 0; part < num_parts; ++part) {
                sum += tuple_subhist[(uint64_t)part * 4 * 256 * 2 + lane_offset + i];
            }
            hist_output[(bin_start + i / 2) * 2 + (i % 2)] = sum;
        }
    }
}

// ===========================================================================
// subtract_histograms — elementwise larger = parent - smaller
// ===========================================================================

kernel void subtract_histograms(
    const device float* parent_hist [[buffer(0)]],
    const device float* smaller_hist [[buffer(1)]],
    device float* larger_hist [[buffer(2)]],
    constant uint& num_items [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_items) return;
    larger_hist[gid] = parent_hist[gid] - smaller_hist[gid];
}

// ===========================================================================
// histogram_private — ZERO atomics, private histogram per thread
//
// The key insight: 32KB threadgroup memory fits 16 private histograms
// (16 × 256 bins × 2 × 4 bytes = 32KB). Each thread accumulates into
// its own histogram with DIRECT WRITES — no atomics, no contention.
// After accumulation, merge 16 histograms (trivial reduction).
//
// This replicates what CPU does: each core has a private histogram.
// Low thread count (16) is compensated by ~10x fewer cycles per write.
// ===========================================================================

kernel void histogram_private(
    const device float*   ordered_grad     [[buffer(0)]],
    const device float*   ordered_hess     [[buffer(1)]],
    const device uchar*   gathered_bins    [[buffer(2)]],
    const device uint*    group_bin_offsets [[buffer(3)]],
    device float*         hist_output      [[buffer(4)]],
    constant uint&        num_data_in_leaf [[buffer(5)]],
    constant uint&        num_groups       [[buffer(6)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    const uint grp = gid;
    if (grp >= num_groups) return;

    const uint bin_start = group_bin_offsets[grp];
    const uint nbins = group_bin_offsets[grp + 1] - bin_start;

    // 16 private histograms in threadgroup memory — 32KB total
    // Layout: [thread_id][bin][grad_or_hess]
    constexpr uint MAX_BINS_X2 = 256 * 2;
    threadgroup float private_hist[16 * MAX_BINS_X2];  // 32KB

    // Zero my private histogram
    const uint my_off = tid * MAX_BINS_X2;
    for (uint i = 0; i < nbins * 2; i++) {
        private_hist[my_off + i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate — DIRECT WRITES, zero contention
    const device uchar* grp_bins = gathered_bins + (uint64_t)grp * num_data_in_leaf;
    for (uint i = tid; i < num_data_in_leaf; i += tgs) {
        const float g = ordered_grad[i];
        const float h = ordered_hess[i];
        const uint bin = grp_bins[i];
        private_hist[my_off + bin * 2]     += g;
        private_hist[my_off + bin * 2 + 1] += h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge: each thread reduces a slice of bins across all private histograms
    for (uint b = tid; b < nbins; b += tgs) {
        float sum_g = 0.0f;
        float sum_h = 0.0f;
        for (uint t = 0; t < tgs; t++) {
            sum_g += private_hist[t * MAX_BINS_X2 + b * 2];
            sum_h += private_hist[t * MAX_BINS_X2 + b * 2 + 1];
        }
        hist_output[(bin_start + b) * 2]     = sum_g;
        hist_output[(bin_start + b) * 2 + 1] = sum_h;
    }
}

// ===========================================================================
// histogram_grouped — fallback for narrow datasets (no gather needed)
//
// One threadgroup per feature group. Uses original ungathered data with
// data_indices indirection. Bin data is column-major [groups × total_rows].
// ===========================================================================

kernel void histogram_grouped(
    const device float*   gradients        [[buffer(0)]],
    const device float*   hessians         [[buffer(1)]],
    const device uchar*   bin_data         [[buffer(2)]],
    const device uint*    group_bin_offsets [[buffer(3)]],
    const device int*     data_indices     [[buffer(4)]],
    device float*         hist_output      [[buffer(5)]],
    constant uint&        num_data_in_leaf [[buffer(6)]],
    constant uint&        num_groups       [[buffer(7)]],
    constant uint&        total_bins       [[buffer(8)]],
    constant uint&        num_data_total   [[buffer(9)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]])
{
    const uint grp = gid;
    if (grp >= num_groups) return;

    const uint bin_start = group_bin_offsets[grp];
    const uint nbins = group_bin_offsets[grp + 1] - bin_start;

    threadgroup atomic_uint local_hist[256 * 2];

    for (uint i = tid; i < nbins * 2; i += tgs) {
        atomic_store_explicit(&local_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device uchar* grp_bins = bin_data + (uint64_t)grp * num_data_total;

    for (uint i = tid; i < num_data_in_leaf; i += tgs) {
        const int data_idx = data_indices[i];
        const float g = gradients[data_idx];
        const float h = hessians[data_idx];
        const uint bin = grp_bins[data_idx];

        atomic_add_float_tg(&local_hist[bin * 2], g);
        atomic_add_float_tg(&local_hist[bin * 2 + 1], h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < nbins * 2; i += tgs) {
        hist_output[(bin_start + i / 2) * 2 + (i % 2)] =
            as_type<float>(atomic_load_explicit(&local_hist[i], memory_order_relaxed));
    }
}

// ===========================================================================
// partition_indices_numeric — partition one leaf for a numerical split
//
// Supported Metal scope only allows dense numerical features, so the split
// rule matches Tree::NumericalDecisionInner on raw bin values:
//   - missing Zero routes by default bin
//   - missing NaN routes by max bin
//   - otherwise compare bin <= threshold
//
// Left rows are packed from the front, right rows from the back. Order within
// each side is not stable, which matches the existing non-stable CPU partition.
// ===========================================================================

kernel void partition_indices_numeric(
    const device uchar*   group_bins        [[buffer(0)]],
    const device int*     input_indices     [[buffer(1)]],
    device int*           output_indices    [[buffer(2)]],
    device atomic_uint*   partition_counts  [[buffer(3)]],  // [left_count, right_count]
    constant uint&        num_data_in_leaf  [[buffer(4)]],
    constant uint&        threshold         [[buffer(5)]],   // CPU split threshold
    constant uint&        default_bin       [[buffer(6)]],
    constant uint&        most_freq_bin     [[buffer(7)]],
    constant uint&        max_bin           [[buffer(8)]],   // stored-bin max
    constant int&         default_left      [[buffer(9)]],
    constant int&         missing_is_zero   [[buffer(10)]],
    constant int&         missing_is_na     [[buffer(11)]],
    constant int&         mfb_is_zero       [[buffer(12)]],
    constant int&         mfb_is_na         [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_data_in_leaf) return;

    const int row = input_indices[gid];
    const uint bin = group_bins[row];
    const uint th = threshold + 1u - (most_freq_bin == 0u ? 1u : 0u);
    const uint t_zero_bin = 1u + default_bin - (most_freq_bin == 0u ? 1u : 0u);
    const bool default_to_left = most_freq_bin <= threshold;
    const bool missing_default_to_left = default_left != 0;
    const bool wide_case = 1u < max_bin;
    bool go_left;

    if (wide_case) {
        if ((missing_is_zero && !mfb_is_zero && bin == t_zero_bin) ||
            (missing_is_na && !mfb_is_na && bin == max_bin)) {
            go_left = missing_default_to_left;
        } else if (bin == 0u) {
            if ((missing_is_na && mfb_is_na) ||
                (missing_is_zero && mfb_is_zero)) {
                go_left = missing_default_to_left;
            } else {
                go_left = default_to_left;
            }
        } else {
            go_left = !(bin > th);
        }
    } else {
        const bool max_bin_to_left = max_bin <= th;
        if (missing_is_zero && !mfb_is_zero && bin == t_zero_bin) {
            go_left = missing_default_to_left;
        } else if (bin != max_bin) {
            if ((missing_is_na && mfb_is_na) ||
                (missing_is_zero && mfb_is_zero)) {
                go_left = missing_default_to_left;
            } else {
                go_left = default_to_left;
            }
        } else if (missing_is_na && !mfb_is_na) {
            go_left = missing_default_to_left;
        } else {
            go_left = max_bin_to_left;
        }
    }

    if (go_left) {
        const uint pos = atomic_fetch_add_explicit(
            &partition_counts[0], 1u, memory_order_relaxed);
        output_indices[pos] = row;
    } else {
        const uint pos = atomic_fetch_add_explicit(
            &partition_counts[1], 1u, memory_order_relaxed);
        output_indices[num_data_in_leaf - 1u - pos] = row;
    }
}

// ===========================================================================
// find_best_split_numeric — one threadgroup per feature-direction task
//
// Mirrors the CUDA path more closely: one thread maps to one histogram bin,
// threadgroup prefix sums produce cumulative left / right statistics, and a
// threadgroup reduction selects the best threshold for that task.
// ===========================================================================

constant float kMetalSplitEpsilon = 1e-15f;
constant float kMetalSplitMinScore = -INFINITY;

struct MetalSplitFindTaskKernel {
    int   inner_feature_index;
    int   reverse;
    int   skip_default_bin;
    int   na_as_missing;
    int   assume_out_default_left;
    uint  hist_offset;
    uint  mfb_offset;
    uint  num_bin;
    uint  default_bin;
};

struct MetalSplitResultKernel {
    float gain;
    int   feature;
    uint  threshold;
    int   default_left;
    float left_sum_gradient;
    float left_sum_hessian;
    int   left_count;
    float right_sum_gradient;
    float right_sum_hessian;
    int   right_count;
    float left_value;
    float right_value;
    int   found;
};

inline float metal_threshold_l1(float sum_gradient, float lambda_l1) {
    const float reg = max(0.0f, abs(sum_gradient) - lambda_l1);
    return sum_gradient >= 0.0f ? reg : -reg;
}

inline float metal_leaf_gain(float sum_gradient, float sum_hessian,
                             float lambda_l1, float lambda_l2) {
    if (lambda_l1 > 0.0f) {
        const float sg = metal_threshold_l1(sum_gradient, lambda_l1);
        return (sg * sg) / (sum_hessian + lambda_l2);
    }
    return (sum_gradient * sum_gradient) / (sum_hessian + lambda_l2);
}

inline float metal_leaf_output(float sum_gradient, float sum_hessian,
                               float lambda_l1, float lambda_l2) {
    if (lambda_l1 > 0.0f) {
        return -metal_threshold_l1(sum_gradient, lambda_l1) / (sum_hessian + lambda_l2);
    }
    return -sum_gradient / (sum_hessian + lambda_l2);
}

inline float metal_split_gain(float left_grad, float left_hess,
                              float right_grad, float right_hess,
                              float lambda_l1, float lambda_l2) {
    return metal_leaf_gain(left_grad, left_hess, lambda_l1, lambda_l2) +
           metal_leaf_gain(right_grad, right_hess, lambda_l1, lambda_l2);
}

kernel void find_best_split_numeric(
    const device MetalSplitFindTaskKernel* tasks                [[buffer(0)]],
    constant uint&                         num_tasks            [[buffer(1)]],
    const device char*                     is_feature_used      [[buffer(2)]],
    const device float*                    histogram            [[buffer(3)]],
    constant float&                        total_gradient       [[buffer(4)]],
    constant float&                        total_hessian_input  [[buffer(5)]],
    constant uint&                         total_count_input    [[buffer(6)]],
    constant float&                        parent_gain          [[buffer(7)]],
    constant float&                        lambda_l1            [[buffer(8)]],
    constant float&                        lambda_l2            [[buffer(9)]],
    constant float&                        min_gain_to_split    [[buffer(10)]],
    constant int&                          min_data_in_leaf     [[buffer(11)]],
    constant float&                        min_sum_hessian_in_leaf [[buffer(12)]],
    device MetalSplitResultKernel*         output_splits        [[buffer(13)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (group_id >= num_tasks) {
        return;
    }

    const device MetalSplitFindTaskKernel& task = tasks[group_id];
    device MetalSplitResultKernel& out = output_splits[group_id];
    const bool feature_used = is_feature_used[task.inner_feature_index] != 0;
    const float total_hessian = total_hessian_input + 2.0f * kMetalSplitEpsilon;
    const int total_count = static_cast<int>(total_count_input);
    const float cnt_factor = float(total_count_input) / total_hessian;
    const float min_gain_shift = parent_gain + min_gain_to_split;
    const uint offset = task.mfb_offset;
    const uint feature_num_bin_minus_offset = task.num_bin - offset;
    const device float* hist = histogram + (task.hist_offset << 1);

    threadgroup float grad_scan[256];
    threadgroup float hess_scan[256];
    threadgroup float gain_scan[256];
    threadgroup uint best_index_scan[256];
    threadgroup uchar valid_scan[256];

    if (tid == 0) {
        out.gain = kMetalSplitMinScore;
        out.feature = -1;
        out.threshold = 0;
        out.default_left = task.assume_out_default_left;
        out.left_sum_gradient = 0.0f;
        out.left_sum_hessian = 0.0f;
        out.left_count = 0;
        out.right_sum_gradient = 0.0f;
        out.right_sum_hessian = 0.0f;
        out.right_count = 0;
        out.left_value = 0.0f;
        out.right_value = 0.0f;
        out.found = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_grad_hist = 0.0f;
    float local_hess_hist = 0.0f;
    const bool skip_sum = task.reverse
        ? (task.skip_default_bin &&
           (task.num_bin - 1u - tid) == task.default_bin)
        : (task.skip_default_bin &&
           (tid + offset) == task.default_bin);

    if (feature_used) {
        if (task.reverse == 0) {
            if (task.na_as_missing != 0 && offset == 1u) {
                if (tid > 0u && tid < task.num_bin) {
                    const uint bin_offset = (tid - 1u) << 1;
                    local_grad_hist = hist[bin_offset];
                    local_hess_hist = hist[bin_offset + 1u];
                }
            } else if (tid < feature_num_bin_minus_offset && !skip_sum) {
                const uint bin_offset = tid << 1;
                local_grad_hist = hist[bin_offset];
                local_hess_hist = hist[bin_offset + 1u];
            }
        } else if (tid >= uint(task.na_as_missing) &&
                   tid < feature_num_bin_minus_offset && !skip_sum) {
            const uint read_index = feature_num_bin_minus_offset - 1u - tid;
            const uint bin_offset = read_index << 1;
            local_grad_hist = hist[bin_offset];
            local_hess_hist = hist[bin_offset + 1u];
        }
    }

    if (feature_used && task.reverse == 0 &&
        task.na_as_missing != 0 && offset == 1u) {
        grad_scan[tid] = local_grad_hist;
        hess_scan[tid] = local_hess_hist;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                grad_scan[tid] += grad_scan[tid + stride];
                hess_scan[tid] += hess_scan[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            local_grad_hist += (total_gradient - grad_scan[0]);
            local_hess_hist += (total_hessian_input - hess_scan[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        local_hess_hist += kMetalSplitEpsilon;
    }

    grad_scan[tid] = local_grad_hist;
    hess_scan[tid] = local_hess_hist;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        float add_grad = 0.0f;
        float add_hess = 0.0f;
        if (tid >= stride) {
            add_grad = grad_scan[tid - stride];
            add_hess = hess_scan[tid - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= stride) {
            grad_scan[tid] += add_grad;
            hess_scan[tid] += add_hess;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_gain = kMetalSplitMinScore;
    bool threshold_found = false;
    uint threshold_value = 0u;
    float best_left_gradient = 0.0f;
    float best_left_hessian = 0.0f;
    int best_left_count = 0;

    if (feature_used) {
        if (task.reverse != 0) {
            if (tid >= uint(task.na_as_missing) &&
                tid < feature_num_bin_minus_offset &&
                tid <= task.num_bin - 2u && !skip_sum) {
                const float sum_right_gradient = grad_scan[tid];
                const float sum_right_hessian = hess_scan[tid];
                const int right_count = static_cast<int>(rint(sum_right_hessian * cnt_factor));
                const float sum_left_gradient = total_gradient - sum_right_gradient;
                const float sum_left_hessian = total_hessian - sum_right_hessian;
                const int left_count = total_count - right_count;
                if (sum_left_hessian >= min_sum_hessian_in_leaf &&
                    left_count >= min_data_in_leaf &&
                    sum_right_hessian >= min_sum_hessian_in_leaf &&
                    right_count >= min_data_in_leaf) {
                    const float current_gain = metal_split_gain(
                        sum_left_gradient, sum_left_hessian,
                        sum_right_gradient, sum_right_hessian,
                        lambda_l1, lambda_l2);
                    if (current_gain > min_gain_shift) {
                        local_gain = current_gain - min_gain_shift;
                        threshold_value = task.num_bin - 2u - tid;
                        threshold_found = true;
                        best_left_gradient = sum_left_gradient;
                        best_left_hessian = sum_left_hessian;
                        best_left_count = left_count;
                    }
                }
            }
        } else {
            const uint end = (task.na_as_missing != 0 && offset == 1u)
                ? (task.num_bin - 2u)
                : (feature_num_bin_minus_offset - 2u);
            if (tid <= end && !skip_sum) {
                const float sum_left_gradient = grad_scan[tid];
                const float sum_left_hessian = hess_scan[tid];
                const int left_count = static_cast<int>(rint(sum_left_hessian * cnt_factor));
                const float sum_right_gradient = total_gradient - sum_left_gradient;
                const float sum_right_hessian = total_hessian - sum_left_hessian;
                const int right_count = total_count - left_count;
                if (sum_left_hessian >= min_sum_hessian_in_leaf &&
                    left_count >= min_data_in_leaf &&
                    sum_right_hessian >= min_sum_hessian_in_leaf &&
                    right_count >= min_data_in_leaf) {
                    const float current_gain = metal_split_gain(
                        sum_left_gradient, sum_left_hessian,
                        sum_right_gradient, sum_right_hessian,
                        lambda_l1, lambda_l2);
                    if (current_gain > min_gain_shift) {
                        local_gain = current_gain - min_gain_shift;
                        threshold_value = (task.na_as_missing != 0 && offset == 1u)
                            ? tid
                            : (tid + offset);
                        threshold_found = true;
                        best_left_gradient = sum_left_gradient;
                        best_left_hessian = sum_left_hessian;
                        best_left_count = left_count;
                    }
                }
            }
        }
    }

    gain_scan[tid] = local_gain;
    best_index_scan[tid] = tid;
    valid_scan[tid] = threshold_found ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const bool self_valid = valid_scan[tid] != 0;
            const bool other_valid = valid_scan[tid + stride] != 0;
            if ((!self_valid && other_valid) ||
                (self_valid && other_valid &&
                 (gain_scan[tid + stride] > gain_scan[tid] ||
                  (gain_scan[tid + stride] == gain_scan[tid] &&
                   best_index_scan[tid + stride] < best_index_scan[tid])))) {
                gain_scan[tid] = gain_scan[tid + stride];
                best_index_scan[tid] = best_index_scan[tid + stride];
                valid_scan[tid] = valid_scan[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid_scan[0] == 0u) {
        return;
    }

    const uint winning_tid = best_index_scan[0];
    if (tid == winning_tid && threshold_found) {
        const float final_left_hessian = best_left_hessian - kMetalSplitEpsilon;
        const float final_right_gradient = total_gradient - best_left_gradient;
        const float final_right_hessian =
            total_hessian - best_left_hessian - kMetalSplitEpsilon;
        out.gain = local_gain;
        out.feature = task.inner_feature_index;
        out.threshold = threshold_value;
        out.default_left = task.assume_out_default_left;
        out.left_sum_gradient = best_left_gradient;
        out.left_sum_hessian = final_left_hessian;
        out.left_count = best_left_count;
        out.right_sum_gradient = final_right_gradient;
        out.right_sum_hessian = final_right_hessian;
        out.right_count = total_count - best_left_count;
        out.left_value = metal_leaf_output(best_left_gradient, final_left_hessian,
                                           lambda_l1, lambda_l2);
        out.right_value = metal_leaf_output(final_right_gradient, final_right_hessian,
                                            lambda_l1, lambda_l2);
        out.found = 1;
    }
}
