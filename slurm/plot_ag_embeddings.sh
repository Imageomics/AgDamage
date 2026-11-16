#!/bin/bash
# CPU projection/report job. Arguments are forwarded to plot_embeddings.py:
#   sbatch --cluster=<cluster> --account=<account> --partition=debug slurm/plot_ag_embeddings.sh \
#     --dataset Waterhemp_GreenHouse_Images_cropped \
#     --thumbnails data/thumbnails_cropped.json \
#     --out reports/embedding_projections_cropped.html
# Pass your site's account at submission time, for example: --account=<account>.
#SBATCH --job-name=agdamage_plot
#SBATCH --partition=serial
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/plot_%j.log
set -e
cd "$SLURM_SUBMIT_DIR"

PY=.venv/bin/python
[ -x "$PY" ] || { echo "missing venv: $PY" >&2; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "===== cluster: $SLURM_CLUSTER_NAME | node: $(hostname) | venv: $PY ====="
$PY scripts/plot_embeddings.py "$@"
echo "===== DONE ====="
