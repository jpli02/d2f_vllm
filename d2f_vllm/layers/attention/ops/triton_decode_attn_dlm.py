# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# type: ignore

# Adapted from vllm
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_decode_attention.py
# formerly adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# which was originally adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

# Changes:
# - Add support for page size >= 1.

# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size >= 1.
"""

import torch
import logging

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

is_hip_ = current_platform.is_rocm()

logger = logging.getLogger(__name__)

# Only print the following warnings when triton version < 3.2.0.
# The issue won't affect performance or accuracy.
if triton.__version__ < '3.2.0':
    logger.warning(
        "The following error message 'operation scheduled before its operands' "
        "can be ignored.")


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    q,
    k_cache,
    v_cache,
    softmax_scale,
    block_tables,
    cache_seqlens,
    attn_logits,
    stride_block_tables_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(cache_seqlens + cur_batch)
    cur_batch_req_idx = cur_batch

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q_vec = tl.load(q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                              cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                block_tables + stride_block_tables_b * cur_batch_req_idx +
                offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (kv_loc[:, None] * stride_buf_kbs +
                          cur_kv_head * stride_buf_kh + offs_d[None, :])
            k = tl.load(
                k_cache + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q_vec[None, :] * k, 1)
            qk *= softmax_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (kv_loc[:, None] * stride_buf_vbs +
                          cur_kv_head * stride_buf_vh + offs_dv[None, :])
            v = tl.load(
                v_cache + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh +
                      split_kv_id * stride_mid_os + offs_dv)

        tl.store(
            attn_logits + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh +
                        split_kv_id * stride_mid_os + Lv)

        tl.store(
            attn_logits + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_attn_m_fwd(
    q,
    k_cache,
    v_cache,
    attn_logits,
    block_tables,
    cache_seqlens,
    num_kv_splits,
    softmax_scale,
    page_size,
    logit_cap,
):
    BLOCK = 64 if not is_hip_ else 8

    NUM_KV_SPLITS = num_kv_splits
    Lk = k_cache.shape[-1]
    Lv = v_cache.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_cache.shape[-2]

    num_warps = 4
    if kv_group_num != 1:
        num_warps = 1 if is_hip_ else 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_cache,
        v_cache,
        softmax_scale,
        block_tables,
        cache_seqlens,
        attn_logits,
        block_tables.stride(0),
        q.stride(0),
        q.stride(1),
        k_cache.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_cache.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_cache.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_cache.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    q,
    k_cache,
    v_cache,
    softmax_scale,
    block_tables,
    cache_seqlens,
    attn_logits,
    stride_block_tables_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(cache_seqlens + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q_vec = tl.load(q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :])
        qpe = tl.load(q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                block_tables + stride_block_tables_b * cur_batch_req_idx +
                offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None])
            k = tl.load(
                k_cache + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q_vec, k.to(q_vec.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[:, None])
                kpe = tl.load(
                    k_cache + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) &
                    (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= softmax_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                          qk, float("-inf"))

            offs_buf_v = (kv_loc[:, None] * stride_buf_vbs +
                          cur_kv_head * stride_buf_vh + offs_dv[None, :])
            v = tl.load(
                v_cache + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob +
                      cur_head[:, None] * stride_mid_oh +
                      split_kv_id * stride_mid_os + offs_dv[None, :])

        tl.store(
            attn_logits + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh +
                        split_kv_id * stride_mid_os + Lv)

        tl.store(
            attn_logits + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_attn_m_fwd(
    q,
    k_cache,
    v_cache,
    attn_logits,
    block_tables,
    cache_seqlens,
    diffusion_blk_sz,
    num_kv_splits,
    softmax_scale,
    page_size,
    logit_cap,
):
    BLOCK = 32
    Lk = k_cache.shape[-1]
    Lv = v_cache.shape[-1]

    if is_hip_ and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    num_tokens, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_cache.shape[-2]
    batch_size = num_tokens // diffusion_blk_sz

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    
    grid = (
        batch_size,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        extra_kargs = {
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 2
        }
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_cache,
        v_cache,
        softmax_scale,
        block_tables,
        cache_seqlens,
        attn_logits,
        block_tables.stride(0),
        q.stride(0),
        q.stride(1),
        k_cache.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_cache.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_cache.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_cache.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    attn_logits,
    o,
    cache_seqlens,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(cache_seqlens + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                                  cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(attn_logits + offs_v + split_kv_id * stride_mid_os,
                         mask=mask_d,
                         other=0.0)
            tlogic = tl.load(attn_logits + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    attn_logits,
    q,
    o,
    v_cache,
    cache_seqlens,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_cache.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {
            "waves_per_eu": 4,
            "matrix_instr_nonkdim": 16,
            "kpack": 2
        }

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        attn_logits,
        o,
        cache_seqlens,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_cache,
    v_cache,
    o,
    block_tables,
    cache_seqlens,
    attn_logits,
    num_kv_splits,
    softmax_scale,
    page_size,
    logit_cap=0.0,
):
    _decode_attn_m_fwd(
        q,
        k_cache,
        v_cache,
        attn_logits,
        block_tables,
        cache_seqlens,
        num_kv_splits,
        softmax_scale,
        page_size,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_cache, cache_seqlens,
                                num_kv_splits)


def decode_attention_fwd_grouped(
    q,
    k_cache,
    v_cache,
    o,
    block_tables,
    cache_seqlens,
    diffusion_blk_sz,
    attn_logits,
    num_kv_splits,
    softmax_scale,
    page_size,
    logit_cap=0.0,
):
    _decode_grouped_attn_m_fwd(
        q,
        k_cache,
        v_cache,
        attn_logits,
        block_tables,
        cache_seqlens,
        diffusion_blk_sz,
        num_kv_splits,
        softmax_scale,
        page_size,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits, 
        q, 
        o, 
        v_cache, 
        cache_seqlens, 
        num_kv_splits
    )


def diffusion_lm_decode_attention_fwd(
    q,
    k_cache,
    v_cache,
    block_tables,
    cache_seqlens,
    diffusion_blk_sz,
    o=None,
    attn_logits=None,
    softmax_scale=None,
    num_kv_splits=1,
    page_size=1,
    logit_cap=0.0,
):
    """
    Forward pass for decode attention using Triton kernels.
    
    Args:
        q: Query tensor of shape [batch_size, num_heads, head_dim]. 
           Contains the query vectors for the current decoding step.
        k_cache: Key cache tensor storing all previous key vectors.
                Shape depends on page_size but generally [..., page_size, num_kv_heads, head_dim].
        v_cache: Value cache tensor storing all previous value vectors.
                Shape depends on page_size but generally [..., page_size, num_kv_heads, head_dim].
        o: Output tensor of shape [batch_size, num_heads, head_dim].
           Will store the computed attention output.
        block_tables: Token mapping tensor that maps request indices to token positions
                     in the paged memory layout. Shape [batch_size, max_seq_len // page_size].
        cache_seqlens: Batch sequence lengths tensor of shape [batch_size].
                  Contains the actual sequence length for each batch item.
        attn_logits: Intermediate attention logits tensor used for computation splits.
                    Shape [batch_size, num_heads, num_kv_splits, head_dim + 1].
                    The extra "+1" dimension stores log-sum-exp values (e_max + log(e_sum)) 
                    at index head_dim, while indices 0:head_dim store the attention outputs 
                    for each split. This is needed for numerically stable softmax reduction 
                    across splits in the second stage.
        num_kv_splits: Number of splits for KV cache processing to manage memory usage.
                      Higher values reduce memory but may increase computation overhead.
        softmax_scale: Scaling factor applied to attention scores before softmax.
                      Typically 1/sqrt(head_dim) for scaled dot-product attention.
        page_size: Size of each page in the paged attention memory layout. Default is 1.
                  Larger page sizes can improve memory efficiency.
        logit_cap: Optional logit capping value. If > 0, applies tanh-based capping to
                  attention logits to prevent overflow. Default is 0.0 (no capping).
    """
    kv_group_num = q.shape[1] // v_cache.shape[-2]
    
    o = o if o is not None else torch.empty_like(q).to(q.device, q.dtype)
    batch_size, num_heads, head_dim = q.shape # In CausalLM: batch_size = num_seqs
    attn_logits_shape = (batch_size, num_heads, num_kv_splits, head_dim + 1)
    attn_logits = attn_logits if attn_logits is not None else torch.empty(attn_logits_shape).to(q.device, q.dtype)
    softmax_scale = q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
    assert num_kv_splits == attn_logits.shape[2]
    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_cache,
            v_cache,
            o,
            block_tables,
            cache_seqlens,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            page_size,
            logit_cap,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_cache,
            v_cache,
            o,
            block_tables,
            cache_seqlens,
            diffusion_blk_sz,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            page_size,
            logit_cap,
        )
    return o