#!/bin/bash
working_dir="/scratch/yota/LLM-encoding-model-FU"
data_dir_path="/scratch/yota/LLM-encoding-model-FU/data"
model_representation_type="last_hidden_state"
model_representation_summary="mean"
num_sentences_per_batch=100

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

caption_data_dir="/scratch/yota/LLM-encoding-model-FU/data/nsd_captions"
caption_file_names=("$caption_data_dir"/*.pkl)
remove_file="/scratch/yota/LLM-encoding-model-FU/data/nsd_captions/subj01_captions_train.pkl"
# Rebuild the array excluding the file you want to remove
filtered_caption_file_names=()
for file in "${caption_file_names[@]}"; do
    if [[ "$file" != "$remove_file" ]]; then
        filtered_caption_file_names+=("$file")
    fi
done
# Overwrite the original array
caption_file_names=("${filtered_caption_file_names[@]}")

# loop over caption files
for caption_file_name in "${caption_file_names[@]}"; do
    caption_file_name=$(basename "$caption_file_name")
    echo $caption_file_name
    # loop over model name 
    for model_name in "${model_names[@]}"; do
        echo $model_name
	# loop over checkpoints
        for checkpoint in "${list_checkpoints_as_step[@]}"; do
            # submit the corresponding job            
            echo $checkpoint
	    echo "----"
	    sbatch --export=model_name=$model_name,checkpoint=$checkpoint,model_representation_type=$model_representation_type,model_representation_summary=$model_representation_summary,caption_file_name=$caption_file_name,num_sentences_per_batch=$num_sentences_per_batch,data_dir_path=$data_dir_path,working_dir=$working_dir ./submit_run_single_model_extraction.sh
        done
    done
done
