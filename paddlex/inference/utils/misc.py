# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ...utils.device import get_default_device, parse_device
from ...utils.env import get_device_type


def is_mkldnn_available():
    # XXX: Not sure if this is the best way to check if MKL-DNN is available
    from paddle.inference import Config

    return hasattr(Config, "set_mkldnn_cache_capacity")


_rocm_attention_patched = False

def _apply_rocm_attention_patch():
    """Apply ROCm-safe attention patch for bf16 support."""
    global _rocm_attention_patched
    if _rocm_attention_patched:
        return
    _rocm_attention_patched = True
    
    try:
        from ..models.doc_vlm.modeling.paddleocr_vl._rocm_attention import patch_ernie_attention_for_rocm
        patch_ernie_attention_for_rocm()
    except Exception as e:
        import warnings
        warnings.warn(f"[ROCm] Failed to apply attention patch: {e}")


def is_bfloat16_available(device):
    import paddle
    import paddle.amp
    import os

    # ROCm: bf16 support requires fp32 softmax workaround
    # Set PADDLEX_ROCM_ENABLE_BF16=1 to enable bf16 with ROCm-safe attention
    if paddle.is_compiled_with_rocm():
        enable_bf16 = os.environ.get("PADDLEX_ROCM_ENABLE_BF16", "0") == "1"
        if enable_bf16:
            _apply_rocm_attention_patch()
        return enable_bf16

    if device is None:
        device = get_default_device()
    device_type, _ = parse_device(device)
    return (
        "npu" in get_device_type() or paddle.amp.is_bfloat16_supported()
    ) and device_type in ("gpu", "npu", "xpu", "mlu", "metax_gpu")


def is_float16_available(device):
    import paddle.amp

    if device is None:
        device = get_default_device()
    device_type, _ = parse_device(device)
    return (
        "npu" in get_device_type() or paddle.amp.is_float16_supported()
    ) and device_type in (
        "gpu",
        "npu",
        "xpu",
        "mlu",
        "dcu",
        "metax_gpu",
        "iluvatar_gpu",
    )
