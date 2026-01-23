# PaddleOCR-VL ROCm Deployment Guide

This guide explains how to deploy PaddleOCR-VL (versions 1.0 and 1.5) on AMD ROCm GPUs using PaddleX. It covers environment setup, model configuration, and running inference with both native Paddle backend and vLLM acceleration.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Running PaddleOCR-VL with Native Backend](#running-paddleocr-vl-with-native-backend)
- [Running PaddleOCR-VL with vLLM Backend](#running-paddleocr-vl-with-vllm-backend)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- AMD GPU with ROCm support (tested on ROCm 7.0)
- Python 3.10+
- PaddlePaddle compiled with ROCm support
- vLLM compiled with ROCm support (for vLLM backend)

## Environment Setup

### 1. Install PaddlePaddle for ROCm

Install PaddlePaddle with ROCm support. You can either:

**Option A: Install from pre-built wheel**

```bash
pip install paddlepaddle-rocm -f https://www.paddlepaddle.org.cn/whl/rocm/stable.html
```

**Option B: Build from source**

```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build && cd build
cmake .. -DPY_VERSION=3.10 -DWITH_ROCM=ON -DWITH_TESTING=OFF
make -j$(nproc)
pip install python/dist/paddlepaddle*.whl
```

### 2. Install PaddleX

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
pip install -e ".[ocr]"
```

### 3. Install vLLM for ROCm (Optional, for vLLM backend)

```bash
pip install vllm  # Make sure it's the ROCm version
```

## Model Preparation

### Download Models

Download the required models:

```bash
cd /path/to/PaddleX

# Download layout detection model (PP-DocLayoutV3)
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/internal/tmp/layout_0116.tar
tar -xvf layout_0116.tar

# Download VL model (PaddleOCR-VL-1.5-0.9B)
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/tmp/checkpoint-5000.tar
tar -xvf checkpoint-5000.tar
```

### Generate Pipeline Configuration

```bash
paddlex --get_pipeline_config PaddleOCR-VL-1.5
```

This creates `PaddleOCR-VL-1.5.yaml` in the current directory.

## Running PaddleOCR-VL with Native Backend

### Configuration for Native Backend

Create or modify `PaddleOCR-VL-1.5.yaml`:

```yaml
pipeline_name: PaddleOCR-VL-1.5

batch_size: 64
use_queues: True
use_doc_preprocessor: False
use_layout_detection: True
use_chart_recognition: False
format_block_content: False
merge_layout_blocks: True

SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayoutV3
    model_dir: ./layout_0116  # Path to layout model
    batch_size: 8
    threshold: 0.3
    layout_nms: True

  VLRecognition:
    module_name: vl_recognition
    model_name: PaddleOCR-VL-1.5-0.9B
    model_dir: ./checkpoint-5000  # Path to VL model
    batch_size: 4096
    genai_config:
      backend: native
```

### Run Inference

```bash
cd /path/to/PaddleX
paddlex --pipeline PaddleOCR-VL-1.5.yaml --input your_image.png
```

## Running PaddleOCR-VL with vLLM Backend

Using vLLM provides significant performance improvements for VLM inference.

### Step 1: Start vLLM Server

```bash
cd /path/to/PaddleX

# For PaddleOCR-VL-1.5
paddlex_genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --model_dir ./checkpoint-5000 \
    --backend vllm \
    --host 0.0.0.0 \
    --port 8118

# For PaddleOCR-VL-1.0 (PaddleOCR-VL-0.9B)
paddlex_genai_server \
    --model_name PaddleOCR-VL-0.9B \
    --model_dir ./checkpoint-5000 \
    --backend vllm \
    --host 0.0.0.0 \
    --port 8118
```

Wait for the server to fully start. You should see:
```
INFO: Started server process
INFO: Application startup complete.
```

### Step 2: Verify Server Status

```bash
curl http://localhost:8118/v1/models
```

Expected response:
```json
{"object":"list","data":[{"id":"PaddleOCR-VL-1.5-0.9B","object":"model",...}]}
```

### Step 3: Configure Pipeline for vLLM Backend

Create `PaddleOCR-VL-vllm.yaml`:

```yaml
pipeline_name: PaddleOCR-VL-1.5

batch_size: 64
use_queues: True
use_doc_preprocessor: False
use_layout_detection: True
use_chart_recognition: False
format_block_content: False
merge_layout_blocks: True

SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayoutV3
    model_dir: ./layout_0116
    batch_size: 8
    threshold: 0.3
    layout_nms: True

  VLRecognition:
    module_name: vl_recognition
    model_name: PaddleOCR-VL-1.5-0.9B
    batch_size: 4096
    genai_config:
      backend: vllm-server
      server_url: http://localhost:8118/v1
```

### Step 4: Run Inference

```bash
paddlex --pipeline PaddleOCR-VL-vllm.yaml --input your_image.png
```

## Configuration Reference

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pipeline_name` | Pipeline identifier (`PaddleOCR-VL` or `PaddleOCR-VL-1.5`) | - |
| `use_layout_detection` | Enable layout detection | `True` |
| `use_doc_preprocessor` | Enable document preprocessing | `False` |
| `use_chart_recognition` | Enable chart recognition | `False` |
| `merge_layout_blocks` | Merge adjacent layout blocks | `True` |

### VLRecognition Options

| Parameter | Description |
|-----------|-------------|
| `model_name` | Model identifier (`PaddleOCR-VL-0.9B` or `PaddleOCR-VL-1.5-0.9B`) |
| `model_dir` | Path to model directory (for native backend) |
| `genai_config.backend` | Backend type: `native` or `vllm-server` |
| `genai_config.server_url` | vLLM server URL (for vllm-server backend) |

### vLLM Server Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Model name to serve | Required |
| `--model_dir` | Path to model directory | Required |
| `--backend` | Backend type (`vllm`, `fastdeploy`, `sglang`) | `vllm` |
| `--host` | Server host address | `localhost` |
| `--port` | Server port | `8000` |

## Troubleshooting

### Issue: `fused_conv2d_add_act` kernel not registered

**Symptom:**
```
RuntimeError: (NotFound) The kernel `fused_conv2d_add_act` is not registered.
```

**Solution:** This is a ROCm compatibility issue. The fix is included in this version of PaddleX. If you encounter this error, ensure you're using the latest code that includes the ROCm-specific pass deletion:

```python
# In paddlex/inference/models/common/static_infer.py
if paddle.is_compiled_with_rocm():
    config.delete_pass("conv2d_add_act_fuse_pass")
    config.delete_pass("conv2d_add_fuse_pass")
```

### Issue: GPU Memory Access Fault

**Symptom:**
```
Memory access fault by GPU node-1
```

**Solution:** This is caused by MIOpen bf16 convolution bugs on ROCm. The fix keeps the visual encoder modules in fp32 precision. Ensure the `_keep_in_fp32_modules` attribute is set in the model class:

```python
class PaddleOCRVLForConditionalGeneration(Ernie4_5PretrainedModel):
    _keep_in_fp32_modules = ["visual", "mlp_AR"]
```

### Issue: ImportError when running from subdirectory

**Symptom:**
```
ImportError: cannot import name 'create_pipeline' from 'paddlex'
```

**Solution:** Always run PaddleX commands from the PaddleX root directory, not from subdirectories like `test/`.

```bash
# Correct
cd /path/to/PaddleX
paddlex --pipeline config.yaml --input test/image.png

# Incorrect
cd /path/to/PaddleX/test
paddlex --pipeline ../config.yaml --input image.png
```

### Issue: `inference.yml` not found for HuggingFace models

**Symptom:**
```
FileNotFoundError: No such file or directory: 'checkpoint-5000/inference.yml'
```

**Solution:** This is expected for HuggingFace-format models. The fix gracefully handles missing `inference.yml` files. Ensure you're using the latest code.

## Performance Comparison

| Backend | Typical Latency | Notes |
|---------|----------------|-------|
| Native (Paddle) | ~2-5s per image | Direct Paddle inference |
| vLLM | ~0.5-1s per image | Optimized VLM inference with batching |

vLLM backend is recommended for production deployments due to better throughput and lower latency.

## References

- [PaddleOCR-VL Official Documentation](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddlePaddle ROCm Installation Guide](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/hardware_support/rocm_docs/paddle_install_cn.html)
- [vLLM Documentation](https://docs.vllm.ai/)

