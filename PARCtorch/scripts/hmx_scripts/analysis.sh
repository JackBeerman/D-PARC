#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic      # Account name
#SBATCH -J per_pixel_analysis      # Job name for the per-pixel analysis
#SBATCH -o %x_%j.out               # Standard output file with Job ID
#SBATCH -e %x_%j.err               # Standard error file with Job ID
#SBATCH -p gpu                     # Partition
#SBATCH --gres=gpu:a100:1           # Request one A40 GPU
#SBATCH -t 01:20:00                # Time limit (10 minutes)
#SBATCH -c 2                       # CPU cores
#SBATCH --mem=40G                  # Memory allocation

# ---------------------------
# Environment Setup
# ---------------------------
echo "========================================================"
echo "Job started on $(hostname) at $(date)"
echo "Setting up the environment..."
module purge
module load miniforge cuda/11.4.2
module load gcc/12.4.0
source activate dparc

# Define path to the Python executable from the conda environment
PYTHON_PATH="/home/jtb3sud/.conda/envs/dparc/bin/python"

# Define the full path to your new Python analysis script
SCRIPT_PATH="/home/jtb3sud/PARCtorch/PARCtorch/scripts/hmx_scripts/per_pixel_analysis.py"

# ---------------------------
# Script Configuration
# ---------------------------
echo "Configuring analysis parameters..."

# --- Key parameters to modify for your experiment ---
SAMPLE_INDEX=0
# Specify the single timestep to analyze
TIMESTEP_TO_ANALYZE="0 1 2 3 4 5 6 7 8 9 10 11 12 13"
LAYER_TO_ANALYZE="upBlock_3" # Can be "initial_doubleConv" or "upBlock_3"

# --- File paths ---
# NOTE: EVAL_DATA_PATH should point to a specific .npz file, not a directory.
MODEL_CHECKPOINT="/home/jtb3sud/PARCtorch/PARCtorch/florian_weights/1701_1900seq3/deform_model_epoch_15.pth"
EVAL_DATA_PATH="/project/vil_baek/data/physics/PARCTorch/HMX/test"
MIN_MAX_JSON="/home/jtb3sud/PARCtorch/PARCtorch/data/real_min_max.json"

# --- Output directory ---
# Create a unique output directory for this job to avoid overwriting results
OUTPUT_DIR="/home/jtb3sud/PARCtorch/PARCtorch/scripts/eval_results/hmx/per_pixel_analysis_job_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------
# Run the Analysis Script
# ---------------------------
echo "--------------------------------------------------------"
echo "Starting Python analysis script..."
echo "SLURM Job ID:      ${SLURM_JOB_ID}"
echo "Sample Index:      ${SAMPLE_INDEX}"
echo "Timestep:          ${TIMESTEP_TO_ANALYZE}"
echo "Layer:             ${LAYER_TO_ANALYZE}"
echo "Model:             ${MODEL_CHECKPOINT}"
echo "Output Directory:  ${OUTPUT_DIR}"
echo "--------------------------------------------------------"

# Execute the Python script with the configured parameters
$PYTHON_PATH "${SCRIPT_PATH}" \
  --model_checkpoint "${MODEL_CHECKPOINT}" \
  --eval_data "${EVAL_DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --min_max_path "${MIN_MAX_JSON}" \
  --timestep ${TIMESTEP_TO_ANALYZE} \
  --layer_name "${LAYER_TO_ANALYZE}"

echo "========================================================"
echo "Python script finished."
echo "Job completed at $(date)"
echo "========================================================"

#  --sample_index ${SAMPLE_INDEX} \