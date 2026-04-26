#!/bin/bash
# scripts/run_cnn_ablations.sh
#
# Runs all six CNN ablation steps sequentially.
# Every run uses the same seed, same dataset, same epoch budget (from shared config).
# Results land in ./output/cnn_step{1..6}_*/
#
# Usage:
#   bash scripts/run_cnn_ablations.sh /path/to/breaking_bad_vol.hdf5 [SEED] [BATCH_SIZE] [MAX_EPOCHS]
#
#   SEED        optional (default: 1116, same as GARF baseline)
#   BATCH_SIZE  optional (default: 32; reduce to 4-8 if GPU VRAM is limited)
#   MAX_EPOCHS  optional (default: 500; reduce to 50 for quick ablation)
#
# Examples:
#   # Full run (default):
#   bash scripts/run_cnn_ablations.sh /data/breaking_bad_vol.hdf5
#
#   # Hardware-constrained run (50 epochs, batch 4):
#   bash scripts/run_cnn_ablations.sh /data/breaking_bad_vol.hdf5 1116 4 50
#
#   # Multi-seed replication:
#   bash scripts/run_cnn_ablations.sh /data/breaking_bad_vol.hdf5 42 32 500

set -e   # stop immediately on any failure — avoids silently broken later steps

DATA_ROOT="${1:?Error: provide path to breaking_bad_vol.hdf5 as first argument}"
SEED="${2:-1116}"
BATCH_SIZE="${3:-32}"
MAX_EPOCHS="${4:-500}"

STEPS=(
    cnn_step1_baseline
    cnn_step2_normals
    cnn_step3_splatting
    cnn_step4_bilinear
    cnn_step5_context
    cnn_step6_attention
    cnn_step7_unet
    cnn_step8a_feat_mean
    cnn_step8b_feat_max
    cnn_step8c_feat_concat
    cnn_step8d_simplecnn_feat_mean
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
    echo "Running: ${EXP_NAME}  (seed=${SEED}, batch=${BATCH_SIZE}, epochs=${MAX_EPOCHS})"
    echo "========================================================"

    python train_cnn_segmentation.py \
        experiment="${STEP}" \
        data.data_root="${DATA_ROOT}" \
        seed="${SEED}" \
        experiment_name="${EXP_NAME}" \
        data.batch_size="${BATCH_SIZE}" \
        trainer.max_epochs="${MAX_EPOCHS}"
done

echo ""
echo "All steps complete."
echo "Results in: ./output/cnn_step*/"
echo "Run:  python scripts/collect_cnn_results.py  to build the summary table."
