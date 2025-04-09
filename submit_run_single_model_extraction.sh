#!/bin/bash

#SBATCH --job-name=extract_model_representation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --qos=hiprio
#SBATCH --partition=gpu
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/yota/sbatch_output/extraction_%j.out
#SBATCH --error=/scratch/yota/sbatch_error/extraction_%j.err


echo "model_name: $model_name"
echo "checkpoint: $checkpoint"
echo "model_representation_type: $model_representation_type"
echo "model_representation_summary: $model_representation_summary"
echo "caption_file_name: $caption_file_name"
echo "num_sentences_per_batch: $num_sentences_per_batch"

# Move the the working directory 
cd $working_dir

# Load the necessary modules and activate the virtual environment
module load Python/3.11.5-GCCcore-13.2.0
source ~/.virtualenvs/LLM_NSD_env/bin/activate

# Optional for PyTorch / Huggingface
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# run code 
/usr/bin/time -v python run_single_model_extraction.py \
    $model_name \
    $checkpoint \
    $model_representation_type \
    $model_representation_summary \
    $caption_file_name \
    --num_sentences_per_batch $num_sentences_per_batch \
    --data_dir_path $data_dir_path \
    2>&1 | tee /scratch/yota/sbatch_log/log_${SLURM_JOB_ID}.txt

deactivate 
module unload Python/3.11.5-GCCcore-13.2.0
