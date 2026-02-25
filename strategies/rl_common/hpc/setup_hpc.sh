#!/bin/bash
# ============================================================================
# setup_hpc.sh — One-time HPC environment setup
#
# Run this ONCE after transferring the strategies folder to the HPC.
# It creates a virtual environment and installs all dependencies.
#
# Usage:
#   bash setup_hpc.sh [venv_path]
#
# Default venv: ~/venvs/rl_packing
# ============================================================================

set -euo pipefail

VENV_PATH="${1:-${HOME}/venvs/rl_packing}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

echo "=================================================="
echo "  HPC Environment Setup"
echo "  Virtual env: ${VENV_PATH}"
echo "  Requirements: ${REQ_FILE}"
echo "=================================================="

# Load Python module (adjust for your HPC)
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
module load CUDA/12.1.1 2>/dev/null || true

# Create venv
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at ${VENV_PATH}"
    echo "To recreate, delete it first: rm -rf ${VENV_PATH}"
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate
source "${VENV_PATH}/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA support)
echo "Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    pip install torch torchvision  # Fallback to default

# Install other requirements
echo "Installing requirements..."
pip install -r "$REQ_FILE"

# Verify
echo ""
echo "── Verification ──"
python -c "
import torch
print(f'  Python:    {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:   {torch.__version__}')
print(f'  CUDA:      {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:       {torch.cuda.get_device_name(0)}')
    print(f'  GPU Mem:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import gymnasium; print(f'  Gymnasium: {gymnasium.__version__}')
import numpy; print(f'  NumPy:     {numpy.__version__}')
import matplotlib; print(f'  Matplotlib:{matplotlib.__version__}')
print('  All OK!')
"

echo ""
echo "=================================================="
echo "  Setup complete! Activate with:"
echo "    source ${VENV_PATH}/bin/activate"
echo ""
echo "  Then run training with:"
echo "    bash ${SCRIPT_DIR}/train_all.sh"
echo "=================================================="
