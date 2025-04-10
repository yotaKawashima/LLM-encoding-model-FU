#!/bin/bash

#SBATCH --job-name=encoding_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=200G
#SBATCH --qos=standard
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/yota/sbatch_output/encoding_%j.out
#SBATCH --error=/scratch/yota/sbatch_error/encoding_%j.err

echo "subject_id: $subject_id"
echo "model_name: $model_name"
echo "checkpoint: $checkpoint"
echo "model_representation_type: $model_representation_type"
echo "model_representation_summary: $model_representation_summary"
echo "use_test_data_for_training: $use_test_data_for_training"

# Move the the working directory 
cd $working_dir

# Load the necessary modules and activate the virtual environment
module load Python/3.11.5-GCCcore-13.2.0
source ~/.virtualenvs/LLM_NSD_env/bin/activate

# run code 
if [ "$use_test_data_for_training" = true ]; then
/usr/bin/time -v python run_single_encoding_model.py \
    $subject_id \
    $model_name \
    $checkpoint \
    $model_representation_type \
    $model_representation_summary \
    --data_dir_path $data_dir_path \
    --use_test_data_for_training \
    2>&1 | tee /scratch/yota/sbatch_log/log_${SLURM_JOB_ID}.txt

else
/usr/bin/time -v python run_single_encoding_model.py \
    $subject_id \
    $model_name \
    $checkpoint \
    $model_representation_type \
    $model_representation_summary \
    --data_dir_path $data_dir_path \
    2>&1 | tee /scratch/yota/sbatch_log/encoding_${SLURM_JOB_ID}.txt
fi

deactivate 
module unload Python/3.11.5-GCCcore-13.2.0

