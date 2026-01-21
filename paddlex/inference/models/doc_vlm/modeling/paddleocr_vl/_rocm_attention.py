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
ROCm-safe attention implementation for bf16 support.
Uses fp32 softmax to avoid MIOpen bf16 softmax limitation.
"""

import paddle
import paddle.nn.functional as F


def rocm_safe_attention(q, k, v, attention_mask=None, sm_scale=None, dropout_p=0.0):
    """
    ROCm-safe attention that computes softmax in fp32.
    
    This avoids the MIOpen bf16 softmax limitation by:
    1. Computing QK^T in the input dtype
    2. Casting to fp32 for softmax
    3. Casting back for the final matmul with V
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]  
        v: Value tensor [batch, heads, seq_len, head_dim]
        attention_mask: Optional attention mask
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
        dropout_p: Dropout probability (default: 0.0)
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    orig_dtype = q.dtype
    head_dim = q.shape[-1]
    
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Compute QK^T - can stay in bf16 for this matmul
    scores = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * sm_scale
    
    # Cast to fp32 for softmax (key step to avoid MIOpen bf16 limitation)
    scores = scores.cast('float32')
    
    # Apply attention mask if provided
    if attention_mask is not None:
        attention_mask = attention_mask.cast('float32')
        scores = scores + attention_mask
    
    # Softmax in fp32
    attn_weights = F.softmax(scores, axis=-1)
    
    # Optional dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, dropout_p)
    
    # Cast weights back to original dtype for final matmul
    attn_weights = attn_weights.cast(orig_dtype)
    
    # Final matmul with V
    out = paddle.matmul(attn_weights, v)
    
    return out


def patch_ernie_attention_for_rocm():
    """
    Patch ERNIE attention to use ROCm-safe attention.
    Call this function before loading the model on ROCm.
    """
    import paddle
    if not paddle.is_compiled_with_rocm():
        return
    
    try:
        from . import _ernie
        
        # Save original core_attn
        original_core_attn = _ernie.Ernie4_5Attention.core_attn
        
        def patched_core_attn(self, q, k, v, attention_mask=None, 
                             attn_mask_start_row_indices=None, seq_length=None):
            """Patched core attention using ROCm-safe implementation."""
            # Reshape for attention: [b, s, h, d] -> [b, h, s, d]
            perm = [0, 2, 1, 3]
            origin_dtype = q.dtype
            
            q = paddle.transpose(q, perm=perm)
            k = paddle.transpose(k, perm=perm)
            v = paddle.transpose(v, perm=perm)
            
            # Handle GQA (grouped query attention)
            replicate = self.config.num_attention_heads // self.config.num_key_value_heads
            if replicate > 1:
                k = paddle.repeat_interleave(k, replicate, axis=1)
                v = paddle.repeat_interleave(v, replicate, axis=1)
            
            # Use ROCm-safe attention
            scale_qk_coeff = self.config.scale_qk_coeff * self.head_dim**0.5
            sm_scale = 1.0 / scale_qk_coeff
            
            out = rocm_safe_attention(q, k, v, attention_mask, sm_scale)
            
            # Reshape back: [b, h, s, d] -> [b, s, h*d]
            out = paddle.transpose(out, perm=[0, 2, 1, 3])
            out = paddle.reshape(out, [0, 0, -1])
            
            # Return output and None for weights (not needed for inference)
            return out, None
        
        # Apply patch
        _ernie.Ernie4_5Attention.core_attn = patched_core_attn
        print("[ROCm] Patched ERNIE attention to use fp32 softmax")
        
    except Exception as e:
        print(f"[ROCm] Warning: Failed to patch ERNIE attention: {e}")


if __name__ == "__main__":
    import paddle
    paddle.device.set_device('gpu')
    
    print("Testing ROCm-safe attention with bf16...")
    
    batch, heads, seq_len, head_dim = 1, 8, 64, 64
    
    q = paddle.randn([batch, heads, seq_len, head_dim], dtype='bfloat16')
    k = paddle.randn([batch, heads, seq_len, head_dim], dtype='bfloat16')
    v = paddle.randn([batch, heads, seq_len, head_dim], dtype='bfloat16')
    
    print(f"Input: q={q.shape}, dtype={q.dtype}")
    
    out = rocm_safe_attention(q, k, v)
    print(f"Output: {out.shape}, {out.dtype}")
    print("✅ ROCm-safe attention bf16 test passed!")

