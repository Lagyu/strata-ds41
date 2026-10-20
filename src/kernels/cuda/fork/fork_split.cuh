// src/kernels/cuda/fork/fork_split.cuh - the fork's split formats for the multi-entry kernels (row_dot_multi).
// Included by iq_kernels.cu right after the Split / kSplit primary templates (fork_fmt.cuh comes before them).
#pragma once

// MXFP4 on the multi-entry kernels (row_dot_multi: a weight part decoded once, dotted with up to GRP_NC activations):
// the two nibble words of a part through kvalues_mxfp4 (the e2m1 values doubled, hence the 0.5 in the scale) and the
// block's e8m0 scale.  apply == vec_dot_mxfp4_q8_1's arithmetic in the same order, so a row is bit for bit the
// single-entry kernel's.  Before this, type 39 took native_gu_kernel / native_down_kernel, which re-read an expert's
// rows once per routed row of a pass (6-16 rows a pass with DSpark).
template<> inline constexpr bool kSplit<39> = true;
template<> struct Split<39> {
    struct W { int2 v[VDR_MXFP4]; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_mxfp4* bq4 = (const block_mxfp4*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < VDR_MXFP4; ++l) r.v[l] = get_int_from_table_16(get_int_b1(bq4->qs, iqs + l), kvalues_mxfp4);
        r.dw = e8m0_to_fp32(bq4->e) * 0.5f;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < VDR_MXFP4; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        return r.dw * __low2float(bq8_1->ds) * sumi;
    }
};
