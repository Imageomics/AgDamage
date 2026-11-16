#!/bin/bash
# Portable GPU job — BioCLIP / BioCLIP 2 / BioCLIP 2.5 / DINOv3 embeddings for the
# images. One array task per model. Submit the SAME script to any OSC cluster
# by changing only --cluster:
#   sbatch --cluster=<cluster> --account=<account> slurm/embed_ag_images.sh
# For a short debug run, add --partition (names differ per cluster) or use submit.sh:
#   sbatch --cluster=<cluster> --account=<account> --partition=<debug-partition> slurm/embed_ag_images.sh
# One model only:
#   sbatch --cluster=<cluster> --account=<account> --partition=<debug-partition> --array=3 slurm/embed_ag_images.sh
# Pass your site's account at submission time, for example: --account=<account>.
#SBATCH --job-name=agdamage_embed
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm/logs/embed_%A_%a.log
set -e
cd "$SLURM_SUBMIT_DIR"

MODELS=(bioclip bioclip-2 bioclip-2.5 dinov3-vit7b16)
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}

# --- pick the venv from whichever cluster this job actually landed on ---
# Slurm sets SLURM_CLUSTER_NAME inside the job; no need to hardcode it per script.
case "$SLURM_CLUSTER_NAME" in
  pitzer)  PY=.venv-cu12/bin/python ;;  # V100 (sm_70): needs the cu124 build (cu130 dropped Volta)
  *)       PY=.venv/bin/python ;;       # cardinal H100 (sm_90) / ascend A100 (sm_80): cu130 build
esac
[ -x "$PY" ] || { echo "missing venv: $PY (see README)" >&2; exit 1; }

# HF_HOME and HF_TOKEN, when needed, are inherited from the submission shell.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "===== cluster: $SLURM_CLUSTER_NAME | node: $(hostname) | model: $MODEL | venv: $PY ====="
nvidia-smi --query-gpu=name --format=csv,noheader

$PY scripts/compute_embeddings.py --model "$MODEL" "$@"

echo "===== DONE ====="
