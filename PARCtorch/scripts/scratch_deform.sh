#!/usr/bin/env bash

#SBATCH -A sds_baek_energetic  # Account name
#SBATCH -J phong_2000  # Job name
#SBATCH -o %x.out              # Standard output
#SBATCH -e %x.err              # Standard error
#SBATCH -p gpu                 # Partition
#SBATCH --gres=gpu:a100:1      # Request GPU
#SBATCH --constraint=a100_80gb
#SBATCH -t 72:00:00            # Time limit
#SBATCH -c 1                   # CPU cores
#SBATCH --mem=80G              # Memory

# Load modules (if needed)
module purge
module load miniforge cuda/11.8.0
source activate dparc


# Use the same Python as JupyterLab
PYTHON_PATH="/home/jtb3sud/.conda/envs/dparc/bin/python"

# Set paths to ensure access to MMCV and other installed packages
export PYTHONPATH="/home/jtb3sud/.conda/envs/dparc/lib/python3.11/site-packages"

# Run your training script
$PYTHON_PATH em_slurm_seq.py \
  --train_dirs "/project/vil_baek/data/physics/PARCTorch/HMX/train" \
  --min_max_output "../data/hmx_min_max.json" \
  --batch_size 2 \
  --num_epochs 600 \
  --load_mod "/home/jtb3sud/PARCtorch/PARCtorch/dparc_em/weights/cr_1600/deform_model_epoch_800.pth" \
  --save_dir "../dparc_em/weights/cr_1600/cr_2000" \
  --loss_dir "../dparc_em/loss/cr_1600/cr_2000" \
  --log_file training.log \
  --device cuda


#  --load_mod "/home/jtb3sud/PARCtorch/PARCtorch/florian_weights\0_800/1401_1700seq2/deform_model_epoch_300.pth" \
#  --load_mod "/home/jtb3sud/PARCtorch/PARCtorch/dparc_em/weights/deform_model_epoch_400.pth" \

#/scratch/jtb3sud/processed/train
#/project/vil_baek/data/physics/PARCTorch/HMX/train