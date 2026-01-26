#!/bin/bash
# PaddleOCR-VL One-Click Deployment Script
# This script downloads models, extracts them, and starts the vLLM server

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PADDLEX_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PADDLEX_DIR}"

# Model URLs
LAYOUT_MODEL_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/internal/tmp/layout_0116.tar"
VL_MODEL_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/tmp/checkpoint-5000.tar"

# Server Configuration
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8118}"
MODEL_NAME="${MODEL_NAME:-PaddleOCR-VL-1.5-0.9B}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to download and extract model
download_model() {
    local url=$1
    local name=$2
    local target_dir=$3
    
    local filename=$(basename "$url")
    local extract_name="${filename%.tar}"
    
    if [ -d "${target_dir}/${extract_name}" ]; then
        log_info "${name} already exists at ${target_dir}/${extract_name}, skipping download."
        return 0
    fi
    
    log_info "Downloading ${name}..."
    wget -q --show-progress -O "${target_dir}/${filename}" "$url"
    
    log_info "Extracting ${name}..."
    tar -xf "${target_dir}/${filename}" -C "${target_dir}"
    
    log_info "Cleaning up ${filename}..."
    rm -f "${target_dir}/${filename}"
    
    log_info "${name} ready at ${target_dir}/${extract_name}"
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check wget
    if ! command -v wget &> /dev/null; then
        log_error "wget is not installed. Please install it first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed. Please install Python 3.10+."
        exit 1
    fi
    
    # Check paddlex
    if ! python -c "import paddlex" &> /dev/null; then
        log_error "PaddleX is not installed. Please run: pip install -e ."
        exit 1
    fi
    
    # Check vllm
    if ! python -c "import vllm" &> /dev/null; then
        log_warn "vLLM is not installed. Install with: pip install vllm"
    fi
    
    log_info "All dependencies are available."
}

# Function to start vLLM server
start_vllm_server() {
    log_info "Starting vLLM server..."
    log_info "  Model: ${MODEL_NAME}"
    log_info "  Model Dir: ${MODEL_DIR}/checkpoint-5000"
    log_info "  Host: ${VLLM_HOST}"
    log_info "  Port: ${VLLM_PORT}"
    
    cd "${PADDLEX_DIR}"
    
    # Check if server is already running
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        log_warn "vLLM server is already running on port ${VLLM_PORT}"
        return 0
    fi
    
    # Start the server
    nohup paddlex_genai_server \
        --model_name "${MODEL_NAME}" \
        --model_dir ./checkpoint-5000 \
        --backend vllm \
        --host "${VLLM_HOST}" \
        --port "${VLLM_PORT}" \
        > "${PADDLEX_DIR}/vllm_server.log" 2>&1 &
    
    local pid=$!
    echo $pid > "${PADDLEX_DIR}/vllm_server.pid"
    
    log_info "vLLM server started with PID: ${pid}"
    log_info "Log file: ${PADDLEX_DIR}/vllm_server.log"
    
    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    local max_retries=60
    local retry=0
    while [ $retry -lt $max_retries ]; do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            log_info "vLLM server is ready!"
            return 0
        fi
        sleep 5
        retry=$((retry + 1))
        echo -n "."
    done
    echo ""
    
    log_error "Server failed to start within timeout. Check ${PADDLEX_DIR}/vllm_server.log for details."
    exit 1
}

# Function to stop vLLM server
stop_vllm_server() {
    log_info "Stopping vLLM server..."
    
    if [ -f "${PADDLEX_DIR}/vllm_server.pid" ]; then
        local pid=$(cat "${PADDLEX_DIR}/vllm_server.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            log_info "Stopped vLLM server (PID: ${pid})"
        else
            log_warn "Server process not found"
        fi
        rm -f "${PADDLEX_DIR}/vllm_server.pid"
    else
        log_warn "PID file not found. Trying to find and kill the process..."
        pkill -f "paddlex_genai_server" || true
    fi
}

# Function to create pipeline config
create_pipeline_config() {
    local config_file="${PADDLEX_DIR}/PaddleOCR-VL-vllm.yaml"
    
    if [ -f "$config_file" ]; then
        log_info "Pipeline config already exists at ${config_file}"
        return 0
    fi
    
    log_info "Creating pipeline configuration..."
    
    cat > "$config_file" << 'EOF'
pipeline_name: PaddleOCR-VL-1.5

batch_size: 64

use_queues: True

use_doc_preprocessor: False
use_layout_detection: True
use_chart_recognition: False
format_block_content: False
merge_layout_blocks: True
markdown_ignore_labels:
  - number
  - footnote
  - header
  - header_image
  - footer
  - footer_image
  - aside_text

SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayoutV3
    model_dir: ./layout_0116
    batch_size: 8
    threshold: 0.3
    layout_nms: True
    layout_unclip_ratio: [1.0, 1.0]
    layout_merge_bboxes_mode:
      0: "union"
      1: "union"
      2: "union"
      3: "large"
      4: "union"
      5: "large"
      6: "large"
      7: "union"
      8: "union"
      9: "union"
      10: "union"
      11: "union"
      12: "union"
      13: "union"
      14: "union"
      15: "large"
      16: "union"
      17: "large"
      18: "union"
      19: "union"
      20: "union"
      21: "union"
      22: "union"
      23: "union"
      24: "union"
  VLRecognition:
    module_name: vl_recognition
    model_name: PaddleOCR-VL-1.5-0.9B
    batch_size: 4096
    genai_config:
      backend: vllm-server
      server_url: http://localhost:8118/v1

SubPipelines:
  DocPreprocessor:
    pipeline_name: doc_preprocessor
    batch_size: 8
    use_doc_orientation_classify: True
    use_doc_unwarping: True
    SubModules:
      DocOrientationClassify:
        module_name: doc_text_orientation
        model_name: PP-LCNet_x1_0_doc_ori
        model_dir: null
        batch_size: 8
      DocUnwarping:
        module_name: image_unwarping
        model_name: UVDoc
        model_dir: null
EOF

    # Update server URL if port is different
    if [ "${VLLM_PORT}" != "8118" ]; then
        sed -i "s|http://localhost:8118/v1|http://localhost:${VLLM_PORT}/v1|g" "$config_file"
    fi
    
    log_info "Pipeline config created at ${config_file}"
}

# Function to run inference test
run_test() {
    local test_image="${1:-${PADDLEX_DIR}/test/paddleocr_vl_demo.png}"
    
    if [ ! -f "$test_image" ]; then
        log_warn "Test image not found at ${test_image}"
        log_info "Downloading test image..."
        mkdir -p "${PADDLEX_DIR}/test"
        wget -q -O "${PADDLEX_DIR}/test/paddleocr_vl_demo.png" \
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
        test_image="${PADDLEX_DIR}/test/paddleocr_vl_demo.png"
    fi
    
    log_info "Running inference test with ${test_image}..."
    cd "${PADDLEX_DIR}"
    paddlex --pipeline PaddleOCR-VL-vllm.yaml --input "$test_image"
}

# Function to show status
show_status() {
    echo ""
    echo "=========================================="
    echo "  PaddleOCR-VL Deployment Status"
    echo "=========================================="
    echo ""
    
    # Check models
    echo "Models:"
    if [ -d "${MODEL_DIR}/layout_0116" ]; then
        echo "  ✓ Layout Model: ${MODEL_DIR}/layout_0116"
    else
        echo "  ✗ Layout Model: NOT FOUND"
    fi
    
    if [ -d "${MODEL_DIR}/checkpoint-5000" ]; then
        echo "  ✓ VL Model: ${MODEL_DIR}/checkpoint-5000"
    else
        echo "  ✗ VL Model: NOT FOUND"
    fi
    
    echo ""
    
    # Check server
    echo "vLLM Server:"
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "  ✓ Running on http://localhost:${VLLM_PORT}"
        echo "  Models: $(curl -s http://localhost:${VLLM_PORT}/v1/models | python -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "Unable to parse")"
    else
        echo "  ✗ Not running"
    fi
    
    echo ""
    echo "Pipeline Config: ${PADDLEX_DIR}/PaddleOCR-VL-vllm.yaml"
    echo ""
}

# Main function
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        deploy)
            log_info "Starting PaddleOCR-VL deployment..."
            echo ""
            check_dependencies
            echo ""
            download_model "$LAYOUT_MODEL_URL" "Layout Detection Model" "$MODEL_DIR"
            echo ""
            download_model "$VL_MODEL_URL" "VL Recognition Model" "$MODEL_DIR"
            echo ""
            create_pipeline_config
            echo ""
            start_vllm_server
            echo ""
            show_status
            log_info "Deployment complete!"
            echo ""
            echo "To run inference:"
            echo "  cd ${PADDLEX_DIR}"
            echo "  paddlex --pipeline PaddleOCR-VL-vllm.yaml --input your_image.png"
            ;;
        download)
            log_info "Downloading models only..."
            download_model "$LAYOUT_MODEL_URL" "Layout Detection Model" "$MODEL_DIR"
            download_model "$VL_MODEL_URL" "VL Recognition Model" "$MODEL_DIR"
            ;;
        start)
            create_pipeline_config
            start_vllm_server
            show_status
            ;;
        stop)
            stop_vllm_server
            ;;
        restart)
            stop_vllm_server
            sleep 2
            start_vllm_server
            show_status
            ;;
        status)
            show_status
            ;;
        test)
            run_test "$2"
            ;;
        help|--help|-h)
            echo "PaddleOCR-VL Deployment Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  deploy    Download models and start vLLM server (default)"
            echo "  download  Download models only"
            echo "  start     Start vLLM server"
            echo "  stop      Stop vLLM server"
            echo "  restart   Restart vLLM server"
            echo "  status    Show deployment status"
            echo "  test      Run inference test (optionally specify image path)"
            echo "  help      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  VLLM_HOST    Server host (default: 0.0.0.0)"
            echo "  VLLM_PORT    Server port (default: 8118)"
            echo "  MODEL_NAME   Model name (default: PaddleOCR-VL-1.5-0.9B)"
            echo ""
            echo "Examples:"
            echo "  $0 deploy              # Full deployment"
            echo "  $0 test image.png      # Run inference on image.png"
            echo "  VLLM_PORT=8000 $0 start  # Start server on port 8000"
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Run '$0 help' for usage information."
            exit 1
            ;;
    esac
}

main "$@"

