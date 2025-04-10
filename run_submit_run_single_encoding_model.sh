#!/bin/bash
working_dir="/scratch/yota/LLM-encoding-model-FU"
data_dir_path="/scratch/yota/LLM-encoding-model-FU/data"
model_representation_type="last_hidden_state"
model_representation_summary="mean"
use_test_data_for_training=false

model_names=("EleutherAI/pythia-410m")
# model_names=(\
# "EleutherAI/pythia-410m", \
# "EleutherAI/pythia-1b", \
# "EleutherAI/pythia-1.4b", \
# "EleutherAI/pythia-2.8b", \
# "EleutherAI/pythia-6.9b")

list_checkpoints_as_step=("step143000")
# list_checkpoints_as_step=(\
# "step0", "step1", "step2", "step4", "step8", \
# "step16", "step32", "step64", "step128", \
# "step500", "step1000", "step2000", "step4000", \
# "step8000", "step10000", "step16000", "step20000", \
# "step30000", "step40000", "step50000", "step60000", \
# "step70000", "step80000", "step90000", "step100000", \
# "step110000", "step120000", "step130000", \
# "step140000", "step143000")

subject_ids=(1)
# subject_ids=({1..8})

# loop over caption files
for subject_id in "${subject_ids[@]}"; do
    # loop over model name 
    for model_name in "${model_names[@]}"; do
        # loop over checkpoints
        for checkpoint in "${list_checkpoints_as_step[@]}"; do
            # submit the corresponding job            
            sbatch --export=subject_id=$subject_id,model_name=$model_name,checkpoint=$checkpoint,model_representation_type=$model_representation_type,model_representation_summary=$model_representation_summary,use_test_data_for_training=$use_test_data_for_training,data_dir_path=$data_dir_path,working_dir=$working_dir submit_run_single_encoding_model.sh
        done
    done
done
