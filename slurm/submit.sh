#!/bin/bash
# One-token submit wrapper so you don't have to remember each cluster's debug-partition name.
#   ./slurm/submit.sh cardinal --account=<account>        # -> gpu partition (long)
#   ./slurm/submit.sh cardinal debug --account=<account>  # -> debug
#   ./slurm/submit.sh pitzer debug --account=<account>    # -> gpudebug
# Any extra args after the (optional) "debug" are forwarded to sbatch, e.g. --array=1.
set -e
cluster="${1:?usage: submit.sh <cardinal|pitzer|ascend> [debug] [extra sbatch args...]}"
shift
mode=""; [ "$1" = "debug" ] && { mode="debug"; shift; }

if [ "$mode" = "debug" ]; then
  case "$cluster" in
    pitzer)  part="gpudebug" ;;   # pitzer's short-GPU queue
    *)       part="debug"    ;;   # cardinal & ascend both call it "debug"
  esac
  part_arg="--partition=$part"
else
  part_arg=""   # let embed_ag_images.sh's #SBATCH --partition=gpu apply
fi

set -x
sbatch --cluster="$cluster" $part_arg "$@" "$(dirname "$0")/embed_ag_images.sh"
