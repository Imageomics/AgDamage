#!/bin/bash
# SAM 3 full-resolution plant masks and native-pixel plant-only crops:
#   sbatch --cluster=<cluster> --account=<account> --partition=debug slurm/segment_ag_images.sh --sample-per-class 2
# Arguments after the script path are forwarded to scripts/segment_images.py.
# Pass your site's account at submission time, for example: --account=<account>.
#SBATCH --job-name=agdamage_segment
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm/logs/segment_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

case "$SLURM_CLUSTER_NAME" in
  pitzer)  PY=.venv-cu12/bin/python ;;
  *)       PY=.venv/bin/python ;;
esac
[ -x "$PY" ] || { echo "missing venv: $PY" >&2; exit 1; }

# HF_HOME and HF_TOKEN, when needed, are inherited from the submission shell.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "===== cluster: $SLURM_CLUSTER_NAME | node: $(hostname) | venv: $PY ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/segment_images.py "$@"

echo "===== DONE ====="
