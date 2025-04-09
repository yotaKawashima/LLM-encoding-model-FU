import numpy as np 
import os 
## Variables related to data path 
#data_dir_path = '/mnt/c/Users/yota/Documents/BCCN/lab_rotations/Adrien_lab/LLM-encoding-model-FU/data'
data_dir_path = '/scratch/yota/LLM-encoding-model-FU/data'
# caption data
caption_dir_name='nsd_captions'
caption_dir = os.path.join(data_dir_path, caption_dir_name)

# model representation data (e.g., last hidden state of LLM)
model_representation_dir_name='model_representations'
model_representation_dir = os.path.join(data_dir_path, model_representation_dir_name)
model_representation_file_name = 'model_representations'
model_representation_file_extension = '.npy' 

# fmri data
fmri_dir_name = 'nsd_fmri'
fmri_dir = os.path.join(data_dir_path, fmri_dir_name)
fmri_file_name = 'betas_average_fsaverage'
fmri_file_extension ='.npy'

# encoding model data
encoding_model_dir_name = 'encoding_model'
encoding_model_dir = os.path.join(data_dir_path, encoding_model_dir_name)
encoding_model_file_name = 'encoding_model'
encoding_model_file_extension = '.npz'

## information about model reprentations
model_representation_type='last_hidden_state' 
model_representation_summary='mean'
num_sentences_per_batch = 10


## list of models
# model sizes in AlKhamissi et al. 2025. 
model_sizes = ['410m']#, '1b', '1.4b', '2.8b', '6.9b']
model_names = [f'EleutherAI/pythia-{size}' for size in model_sizes]

# checkpoints from Figure in AlKhamissi et al. 2025.
# 1 step (iteration) = ~2M tokens
checkpoints = [0]
checkpoints.extend([2**x for x in range(1, 9)])
checkpoints.extend([(2**x)*1000 for x in range(0, 6)])
checkpoints.extend([(20*x)*1000 for x in range(1, 15)])
checkpoints.append(286000)
checkpoints.sort()
checkpoints_as_step = ['step'+str(int(num / 2)) for num in checkpoints]

# regression hyperparameters
alphas = np.logspace(-3, 3, 7)
num_folds = 5 
rois = ['V1', 'V2', 'V3', 'V4']

