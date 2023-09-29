#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:10:00
#SBATCH --mem=40GB
#SBATCH --mail-user=guru@cs.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_3-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate edl_hw3

mkdir -p /scratch/general/vast/u1419540/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419540/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1419540/huggingface_cache"

OUT_DIR=/scratch/general/vast/u1419540/cs6966/assignment3/out/
mkdir -p ${OUT_DIR}
# VIZ_DIR=/scratch/general/vast/u1419540/cs6966/assignment3/out/visualizations
# mkdir -p $(VIZ_DIR)
python revisualize.py \
    --analysis_dir /scratch/general/vast/u1419540/cs6966/assignment3/out/visualizations \
    --model_checkpoint /scratch/general/vast/u1419540/cs6966/assignment1/models/deberta-v3-base-finetuned-imdb_classification/checkpoint-1667/ \
    --a1_analysis_file /uufs/chpc.utah.edu/common/home/u1419540/explainable_deep_learning/a3_highlighting/selected_outputs.jsonl \
    --output_dir /scratch/general/vast/u1419540/cs6966/assignment3/out/

# python train.py --output_dir ${OUT_DIR} 
