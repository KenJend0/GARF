#!/bin/bash
# scripts/run_cnn_ablations.sh
#
# Runs all six CNN ablation steps sequentially.
# Every run uses the same seed, same dataset, same epoch budget (from shared config).
# Results land in ./output/cnn_step{1..6}_*/
#
# Usage:
#   bash scripts/run_cnn_ablations.sh /path/to/breaking_bad_vol.hdf5 [SEED]
#
#   SEED is optional (default: 1116, same as GARF baseline).
#   When provided, experiment_name gets a _seed{N} suffix so runs are kept separate.
#
# Examples:
#   # Single seed (default):
#   bash scripts/run_cnn_ablations.sh /data/breaking_bad_vol.hdf5
#
#   # Multi-seed replication:
#   bash scripts/run_cnn_ablations.sh /data/breaking_bad_vol.hdf5 42
#   bash scripts/run_cnn_ablations.sh /data/breaking_bad_vol.hdf5 2024
#
#   # Then collect results across seeds:
#   python scripts/collect_cnn_results.py --seeds 1116 42 2024

set -e   # stop immediately on any failure — avoids silently broken later steps

DATA_ROOT="${1:?Error: provide path to breaking_bad_vol.hdf5 as first argument}"
SEED="${2:-1116}"

STEPS=(
    cnn_step1_baseline
    cnn_step2_normals
    cnn_step3_splatting
    cnn_step4_bilinear
    cnn_step5_context
    cnn_step6_attention
)

for STEP in "${STEPS[@]}"; do
    # When a non-default seed is used, append _seed{N} to experiment_name so
    # runs with different seeds land in separate output directories.
    if [ "$SEED" = "1116" ]; then
        EXP_NAME="${STEP}"
    else
        EXP_NAME="${STEP}_seed${SEED}"
    fi

    echo "========================================================"
    echo "Running: ${EXP_NAME}  (seed=${SEED})"
    echo "========================================================"

    python train_cnn_segmentation.py \
        experiment="${STEP}" \
        data.data_root="${DATA_ROOT}" \
        seed="${SEED}" \
        experiment_name="${EXP_NAME}"
done

echo ""
echo "All steps complete."
echo "Results in: ./output/cnn_step*/"
echo "Run:  python scripts/collect_cnn_results.py  to build the summary table."
