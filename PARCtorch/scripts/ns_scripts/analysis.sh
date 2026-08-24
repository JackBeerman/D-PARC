#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic      # Account name (Please verify this is correct)
#SBATCH -J ns_perpixel_analysis    # Job name for the per-pixel analysis
#SBATCH -o %x_%j.out               # Standard output file (e.g., ns_perpixel_analysis_12345.out)
#SBATCH -e %x_%j.err               # Standard error file (e.g., ns_perpixel_analysis_12345.err)
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1           # Request one A40 GPU
#SBATCH -t 01:00:00                # Time limit (HH:MM:SS) - Increased for full dataset analysis


# =================================================================== #
# Environment Setup
# =================================================================== #
echo "========================================================"
echo "Job started on $(hostname) at $(date)"
echo "Setting up the environment..."
module purge
module load miniforge cuda/11.4.2
module load gcc/12.4.0
source activate dparc

# =================================================================== #
# Script Configuration
# =================================================================== #
echo "Configuring analysis parameters..."

# --- ⚙️ USER: Please modify these parameters as needed ---
LAYER_TO_ANALYZE="upBlock_3" # Options: "initial_doubleConv", "upBlock_3"
BATCH_SIZE=2                 # Adjust based on GPU memory; smaller is safer
FUTURE_STEPS=38              # Number of future steps the model was trained on

# --- 📂 USER: Verify these paths are correct ---
# Path to the Python executable from your conda environment
PYTHON_PATH="/home/jtb3sud/.conda/envs/dparc/bin/python"

# Full path to the new per-pixel analysis script
SCRIPT_PATH="/home/jtb3sud/PARCtorch/PARCtorch/scripts/ns_scripts/per_pixel_analysis.py"

# File paths for the Navier-Stokes model and data
BASE_DIR="/home/jtb3sud/PARCtorch/PARCtorch"
MODEL_CHECKPOINT="${BASE_DIR}/e4/3501_4000/weights/ns_model_epoch_500.pth"
EVAL_DATA_DIR="/project/vil_baek/data/physics/PARCTorch/NavierStokes/test"
MIN_MAX_JSON="${BASE_DIR}/data/ns_min_max.json"

# --- Output directory ---
# A unique output directory is created for this job's results
OUTPUT_DIR="/home/jtb3sud/PARCtorch/PARCtorch/scripts/eval_results/ns/perpixel_job_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

# =================================================================== #
# Run the Analysis Script
# =================================================================== #
echo "--------------------------------------------------------"
echo "Starting Python per-pixel analysis script..."
echo "SLURM Job ID:      ${SLURM_JOB_ID}"
echo "Layer:             ${LAYER_TO_ANALYZE}"
echo "Batch Size:        ${BATCH_SIZE}"
echo "Model:             ${MODEL_CHECKPOINT}"
echo "Evaluation Data:   ${EVAL_DATA_DIR}"
echo "Output Directory:  ${OUTPUT_DIR}"
echo "--------------------------------------------------------"

# Execute the Python script with the configured parameters
$PYTHON_PATH "${SCRIPT_PATH}" \
  --model_checkpoint "${MODEL_CHECKPOINT}" \
  --eval_data "${EVAL_DATA_DIR}" \
  --min_max_path "${MIN_MAX_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --layer_name "${LAYER_TO_ANALYZE}" \
  --batch_size ${BATCH_SIZE} \
  --future_steps ${FUTURE_STEPS}

echo "========================================================"
echo "Python script finished."
echo "Job completed at $(date)"
echo "========================================================"