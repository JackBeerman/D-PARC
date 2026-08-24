#!/usr/bin/env bash

#SBATCH -A sds_baek_energetic  # Account name
#SBATCH -J burgers_phong_1400  # Job name
#SBATCH -o %x.out              # Standard output
#SBATCH -e %x.err              # Standard error
#SBATCH -p gpu                 # Partition
#SBATCH --gres=gpu:a40:1      # Request GPU
#SBATCH -t 72:00:00            # Time limit
#SBATCH -c 4                   # CPU cores
#SBATCH --mem=80G              # Memory

# Load modules (if needed)
module purge
module load miniforge cuda/11.4.2
source activate dparc


# Use the same Python as JupyterLab
PYTHON_PATH="/home/jtb3sud/.conda/envs/dparc/bin/python"

# Set paths to ensure access to MMCV and other installed packages
export PYTHONPATH="/home/jtb3sud/.conda/envs/dparc/lib/python3.11/site-packages"


# Run your training script
$PYTHON_PATH burgers_slurm.py \
  --train_dirs "/project/vil_baek/data/physics/PARCTorch/Burgers/train" \
  --test_dirs "/project/vil_baek/data/physics/PARCTorch/Burgers/test" \
  --min_max_output "../data/bur_min_max.json" \
  --load_mod "/scratch/jtb3sud/burger_phong/weights/501_1000/burgers_model_epoch_500.pth" \
  --batch_size 4 \
  --num_epochs 400 \
  --save_dir "/scratch/jtb3sud/burger_phong/weights/1001_1400" \
  --loss_dir "/scratch/jtb3sud/burger_phong/loss/1001_1400" \
  --log_file training_burgers.log \
  --device cuda
  
  
#--load_mod "/home/jtb3sud/PARCtorch/PARCtorch/burgers/0_300/weights/burgers_model_epoch_300.pth" \
