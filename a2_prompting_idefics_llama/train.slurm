#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=8:00:00
#SBATCH --mem=65GB
#SBATCH --mail-user=guru@cs.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_2-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate edl_hw2

mkdir -p /scratch/general/vast/u1419540/huggingface_cache
export TRANSFORMER_CACHE="/scratch/general/vast/u1419540/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1419540/huggingface_cache"

OUT_DIR=/scratch/general/vast/u1419540/cs6966/assignment2/out
mkdir -p ${OUT_DIR}
python assignment_2.py --seed 1419540 --output_dir ${OUT_DIR} 
