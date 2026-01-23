# PaddleOCR-VL ROCm Deployment Guide

This guide explains how to deploy PaddleOCR-VL (versions 1.0 and 1.5) on AMD ROCm GPUs using PaddleX. It covers environment setup, model configuration, and running inference with both native Paddle backend and vLLM acceleration.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Running PaddleOCR-VL with Native Backend](#running-paddleocr-vl-with-native-backend)
- [Running PaddleOCR-VL with vLLM Backend](#running-paddleocr-vl-with-vllm-backend)
- [Configuration Reference](#configuration-reference)

## Prerequisites

- AMD GPU with ROCm support (tested on ROCm 7.0)
- Python 3.10+
- PaddlePaddle compiled with ROCm support
- vLLM compiled with ROCm support (for vLLM backend)

### Verify ROCm Installation

```bash
# Check ROCm version
cat /opt/rocm/.info/version
# Expected: 7.0.0-17483

# Verify GPU detection
rocm-smi
```

## Environment Setup

### 1. Build and Install PaddlePaddle for ROCm

#### Clone the Patched Paddle Repository

```bash
git clone -b dev_amd https://github.com/vivienfanghuagood/Paddle.git
cd Paddle
```

#### Create Python Virtual Environment

```bash
python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --upgrade pip
pip install numpy protobuf pyyaml requests
```

#### Configure Build

```bash
mkdir build && cd build

cmake .. \
    -DPY_VERSION=3.10 \
    -DPYTHON_EXECUTABLE=/opt/venv/bin/python \
    -DWITH_ROCM=ON \
    -DON_INFER=ON \
    -DWITH_TESTING=OFF \
    -DWITH_XBYAK=OFF
```

#### Build Paddle

```bash
# Use appropriate number of parallel jobs based on your system
make -j$(nproc)
```

> **Note:** The build may take 1-2 hours depending on your system.

#### Install Paddle

```bash
pip install python/dist/paddlepaddle_rocm*.whl
```

#### Verify Installation

```bash
python -c "
import paddle
print('Paddle version:', paddle.__version__)
print('ROCm compiled:', paddle.is_compiled_with_rocm())
print('GPU available:', paddle.device.is_compiled_with_cuda())
"
```

Expected output:
```
Paddle version: 0.0.0
ROCm compiled: True
GPU available: True
```

### 2. Install PaddleX

```bash
git clone -b dev_rocm70 https://github.com/vivienfanghuagood/PaddleX.git
cd PaddleX
pip install -e ".[ocr]"
```

### 3. Install vLLM for ROCm (Optional, for vLLM backend)

```bash
pip install vllm  # Make sure it's the ROCm version
```

## Running PaddleOCR-VL with Native Backend

### Generate Pipeline Configuration

```bash
paddlex --get_pipeline_config PaddleOCR-VL-1.5
```

This creates `PaddleOCR-VL-1.5.yaml` in the current directory.

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
