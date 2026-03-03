#!/bin/bash
#SBATCH --job-name=pcl_train
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpu
#SBATCH --nodelist=gpuvm14
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=/vol/bitbucket/as9422/SemEval-2022-Task-4/logs/train_%j.out
#SBATCH --error=/vol/bitbucket/as9422/SemEval-2022-Task-4/logs/train_%j.err

# ============================================================
# PCL Detection Training — DeBERTa-v3-base + Focal Loss
# ============================================================
#
# SETUP (run once before submitting):
#   cd /vol/bitbucket/as9422
#   git clone <your-repo-url> SemEval-2022-Task-4
#   cd SemEval-2022-Task-4
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install torch torchvision transformers datasets accelerate scikit-learn matplotlib seaborn
#
# SUBMIT:
#   sbatch train_pcl.sh
#
# MONITOR:
#   squeue -u as9422
#   tail -f /vol/bitbucket/as9422/SemEval-2022-Task-4/logs/train_<JOBID>.out
# ============================================================

BASE_DIR="/vol/bitbucket/as9422/SemEval-2022-Task-4"

# Setup CUDA
source /vol/cuda/12.0.0/setup.sh

# Activate venv
source "${BASE_DIR}/.venv/bin/activate"

# Create log directory
mkdir -p "${BASE_DIR}/logs"

# Print environment info
echo "=== Environment ==="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "==================="
echo ""

# Run training
cd "${BASE_DIR}"
python src/train.py --base_dir "${BASE_DIR}"
