#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J cnn_paper_fig
#SBATCH -o cnn_paper_fig.out
#SBATCH -e cnn_paper_fig.err
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH -t 00:20:00
#SBATCH -c 4
#SBATCH --mem=40G

echo "================================================================"
echo "D-PARC (CNN) PAPER FIGURES"
echo "================================================================"
echo "Start: $(date)"

module purge
module load miniforge cuda/11.8.0
source activate dparc

PYTHON_PATH="/home/jtb3sud/.conda/envs/dparc/bin/python"
export PYTHONPATH="/home/jtb3sud/.conda/envs/dparc/lib/python3.11/site-packages"

# ============================================================
# PATHS
# ============================================================
BASE_DATA="/standard/sds_baek_energetic/PSAAP - SAGEST/Chord_ShockTube_0.5x0.5mDomain_64x64Cells/different_dt"
TEST_DIR="${BASE_DATA}/cnn_datasets/test"
OUTPUT_DIR="/scratch/jtb3sud/shocktube_comparison/paper_figures/cnn_parc"

MODEL_PATH="/scratch/jtb3sud/new_shocktubedt/weights/large/shocktube_model_epoch_750.pth"
MIN_MAX_PATH="/home/jtb3sud/PARCtorch/PARCtorch/data/shocktube_min_max.json"

# ============================================================
# VALIDATE
# ============================================================
echo ""
for label_path in "Test data:$TEST_DIR" "Model:$MODEL_PATH" "Min-max:$MIN_MAX_PATH"; do
    label="${label_path%%:*}"
    fpath="${label_path#*:}"
    if [ -e "$fpath" ]; then
        echo "  ✓ $label: $fpath"
    else
        echo "  ✗ $label: NOT FOUND — $fpath"
    fi
done
echo "  Output: $OUTPUT_DIR"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model checkpoint not found."
    exit 1
fi

if [ ! -f "$MIN_MAX_PATH" ]; then
    echo "❌ Min-max file not found."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "================================================================"

$PYTHON_PATH paper_figure.py \
    --test_dir "$TEST_DIR" \
    --model_path "$MODEL_PATH" \
    --min_max_path "$MIN_MAX_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sim_indices 0 5 \
    --rollout_steps 40 \
    --dpi 300 \
    --cmap RdBu_r \
    --error_fig \
    --device cuda

echo ""
echo "End: $(date)"
echo "✅ Figures in $OUTPUT_DIR"