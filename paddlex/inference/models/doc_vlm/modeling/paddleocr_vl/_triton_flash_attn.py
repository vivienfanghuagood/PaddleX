# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple Triton FlashAttention for ROCm bf16 support.
This avoids the MIOpen softmax bf16 limitation by fusing attention computation.
"""

import triton
import triton.language as tl
import paddle


@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Flash Attention forward kernel."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Base offsets (cast to int64)
    qkv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    
    # Block indices
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Q block pointer
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, BLOCK_K),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    
    # Load Q
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    
    # K, V block pointers
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(N_CTX, BLOCK_K),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, BLOCK_K),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    
    # Loop over K, V blocks
    for _ in range(0, N_CTX, BLOCK_N):
        # Load K, V blocks
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        
        # Compute QK^T in float32
        qk = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * sm_scale
        
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
        
        # Update accumulator
        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v.to(tl.float32))
        
        m_i = m_new
        l_i = l_new
        
        # Advance block pointers
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Finalize
    acc = acc / l_i[:, None]
    
    # Output block pointer
    O_block_ptr = tl.make_block_ptr(
        base=Out + qkv_offset,
        shape=(N_CTX, BLOCK_K),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))


def triton_flash_attention(q, k, v, sm_scale=None):
    """
    Triton-based Flash Attention for ROCm bf16 support.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    batch, heads, seq_len, head_dim = q.shape
    
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Allocate output
    out = paddle.empty_like(q)
    
    # Get strides (in elements, not bytes)
    stride_qz, stride_qh, stride_qm, stride_qk = (
        q.shape[1] * q.shape[2] * q.shape[3],
        q.shape[2] * q.shape[3],
        q.shape[3],
        1
    )
    stride_kz, stride_kh, stride_kn, stride_kk = (
        k.shape[1] * k.shape[2] * k.shape[3],
        k.shape[2] * k.shape[3],
        k.shape[3],
        1
    )
    stride_vz, stride_vh, stride_vn, stride_vk = (
        v.shape[1] * v.shape[2] * v.shape[3],
        v.shape[2] * v.shape[3],
        v.shape[3],
        1
    )
    stride_oz, stride_oh, stride_om, stride_ok = (
        out.shape[1] * out.shape[2] * out.shape[3],
        out.shape[2] * out.shape[3],
        out.shape[3],
        1
    )
    
    # Block sizes
    BLOCK_M = min(64, seq_len)
    BLOCK_N = min(64, seq_len)
    BLOCK_K = head_dim
    
    # Grid
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    
    # Launch kernel
    _fwd_kernel[grid](
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        batch, heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return out


def triton_flash_attention_varlen(q, k, v, sm_scale=None):
    """
    Flash Attention wrapper that handles variable sequence length.
    Falls back to naive attention if Triton fails.
    """
    try:
        return triton_flash_attention(q, k, v, sm_scale)
    except Exception as e:
        # Fallback to naive attention with fp32 softmax
        import warnings
        warnings.warn(f"Triton FA failed ({e}), falling back to naive attention")
        
        batch, heads, seq_len, head_dim = q.shape
        if sm_scale is None:
            sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Compute in fp32 for softmax
        scores = paddle.matmul(q.cast('float32'), k.cast('float32').transpose([0, 1, 3, 2])) * sm_scale
        attn = paddle.nn.functional.softmax(scores, axis=-1)
        out = paddle.matmul(attn, v.cast('float32'))
        
        return out.cast(q.dtype)


if __name__ == "__main__":
    import paddle
    paddle.device.set_device('gpu')
    
    print("Testing Triton Flash Attention...")
    
    batch, heads, seq_len, head_dim = 1, 8, 64, 64
    
    q = paddle.randn([batch, heads, seq_len, head_dim], dtype='bfloat16')
    k = paddle.randn([batch, heads, seq_len, head_dim], dtype='bfloat16')
    v = paddle.randn([batch, heads, seq_len, head_dim], dtype='bfloat16')
    
    print(f"Input: q={q.shape}, dtype={q.dtype}")
    
    out = triton_flash_attention_varlen(q, k, v)
    print(f"Output: {out.shape}, {out.dtype}")
    
    # Reference
    scale = 1.0 / (head_dim ** 0.5)
    scores = paddle.matmul(q.cast('float32'), k.cast('float32').transpose([0, 1, 3, 2])) * scale
    attn = paddle.nn.functional.softmax(scores, axis=-1)
    out_ref = paddle.matmul(attn, v.cast('float32')).cast('bfloat16')
    
    diff = (out.cast('float32') - out_ref.cast('float32')).abs().max()
    print(f"Max diff vs reference: {diff.item():.6f}")
    print("✅ Test passed!" if diff < 0.1 else "❌ Test failed")
