import sys
import os 
import argparse
import numpy as np
import config
from sklearn.linear_model import RidgeCV
import experiments.encoding_model as encoding_model

def main(): 
    # create parser object    
    parser = argparse.ArgumentParser( \
        description='Performe encoding model analysis.', \
        epilog='By Yota Kawashima')
    
    # arguments
    parser.add_argument('subject_id', \
                        type=int, \
                        help='Subject ID.')
    parser.add_argument('model_name', \
                        type=str, \
                        help='model name.')
    parser.add_argument('checkpoint', \
                        type=str, \
                        help='Checkpoint (step).')
    parser.add_argument('model_representation_type', \
                        type=str, \
                        help='Type of model representation (e.g., last_hidden_state).')
    parser.add_argument('model_representation_summary', \
                        type=str, \
                        help='How to summarise model representation across captions per image (e.g., mean).')
    
    # add default arguments
    parser.add_argument('--use_test_data_for_training', \
                        action='store_true', \
                        help='To check the implementation, run a use the test data set for the training and testing.')
    parser.add_argument('--data_dir_path', \
                        type=str, \
                        default=config.data_dir_path, \
                        help='Directory path for data.')
    parser.add_argument('--fmri_dir_name', \
                        type=str, \
                        default=config.fmri_dir_name, \
                        help='Directory name for caption data.')
    parser.add_argument('--fmri_file_name', \
                        type=str, \
                        default=config.fmri_file_name, \
                        help='File name for fMRI data.')
    parser.add_argument('--fmri_file_extension', \
                        type=str, \
                        default=config.fmri_file_extension, \
                        help='File extension for fMRI data.')
    parser.add_argument('--model_representation_dir_name', \
                        type=str, \
                        default=config.model_representation_dir_name, \
                        help='Directory name for model representation data.')
    parser.add_argument('--model_representation_file_name', \
                        type=str, \
                        default=config.model_representation_file_name, \
                        help='File name for model representation data.')
    parser.add_argument('--model_representation_file_extension', \
                        type=str, \
                        default=config.model_representation_file_extension, \
                        help='File extension for model representation data.')
    parser.add_argument('--encoding_model_dir_name', \
                        type=str, \
                        default=config.encoding_model_dir_name, \
                        help='Directory name for encoding model data.')
    parser.add_argument('--encoding_model_file_name', \
                        type=str, \
                        default=config.encoding_model_file_name, \
                        help='File name for encoding model data.')
    parser.add_argument('--encoding_model_file_extention', \
                        type=str, \
                        default=config.encoding_model_file_extension, \
                        help='File extension for encoding model data.')
    parser.add_argument('--alphas', \
                        type=np.ndarray,
                        default=config.alphas, \
                        help='A set of regularization parameters for Ridge regression.')                        
    parser.add_argument('--num_folds', \
                        type=int, \
                        default=config.num_folds, \
                        help='Number of folds for cross-validation.')

    args = parser.parse_args()

    if args.use_test_data_for_training == True:
        print("use test data for training")
    else:
        print("use training data for training")

    # model representation data path 
    # Note that the test data is shared across all participants.
    model_representation_data_dir_path = \
        os.path.join(args.data_dir_path, args.model_representation_dir_name, 
                     args.model_name + '_' + args.checkpoint,
                     args.model_representation_type + '_summary_type_' + args.model_representation_summary)
    subject_model_representation_file_name = \
        f'subj{args.subject_id:02d}_' + args.model_representation_file_name
    
    model_representation_test_path = \
        os.path.join(model_representation_data_dir_path,
                     args.model_representation_file_name + '_test' + args.model_representation_file_extension)
    
    if args.use_test_data_for_training == True:
        model_representation_train_path = \
            os.path.join(model_representation_data_dir_path,
                         args.model_representation_file_name + '_test' + args.model_representation_file_extension)
    else:
        model_representation_train_path = \
            os.path.join(model_representation_data_dir_path,
                         subject_model_representation_file_name + '_train' + args.model_representation_file_extension)
    
    # fmri data path
    subject_fmri_file_name = \
        f'subj{args.subject_id:02d}_' + args.fmri_file_name
    fmri_test_path = \
        os.path.join(args.data_dir_path, args.fmri_dir_name,
                     subject_fmri_file_name + '_test' + args.fmri_file_extension)
    if args.use_test_data_for_training == True:
        fmri_train_path = \
            os.path.join(args.data_dir_path, args.fmri_dir_name,
                         subject_fmri_file_name + '_test' + args.fmri_file_extension)
    else:
        fmri_train_path = \
            os.path.join(args.data_dir_path, args.fmri_dir_name,
                         subject_fmri_file_name + '_train' + args.fmri_file_extension)
                          
    # create the data path to encoding model data
    encoding_model_data_dir_path = \
        os.path.join(args.data_dir_path, args.encoding_model_dir_name,
                     args.model_name + '_' + args.checkpoint,
                     args.model_representation_type + '_summary_type_' + args.model_representation_summary)
    os.makedirs(encoding_model_data_dir_path, exist_ok=True)
    encoding_model_file_name = \
        subject_model_representation_file_name.replace(args.model_representation_file_name, \
                                                       args.encoding_model_file_name)
    if args.use_test_data_for_training == True:
        encoding_model_file_name = \
            'TEST_' + encoding_model_file_name + args.encoding_model_file_extention
    else: 
        encoding_model_file_name = \
            encoding_model_file_name + args.encoding_model_file_extention
        

    encoding_model_data_path = \
        os.path.join(encoding_model_data_dir_path, encoding_model_file_name)
    
    # fit/predict 
    corrs, coeff_determination, best_alpha, best_score_train, _ = \
        encoding_model.fit_and_predict(fmri_train_path, 
                                       fmri_test_path,
                                       model_representation_train_path,
                                       model_representation_test_path,
                                       regression=RidgeCV(alphas=args.alphas, cv=args.num_folds))
    
    # save file 
    encoding_model.save_prediction_in_single_file(encoding_model_data_path, \
                                                  corrs=corrs, 
                                                  coeff_determination=coeff_determination,
                                                  best_alpha=best_alpha,
                                                  best_score_train=best_score_train)

if __name__ == '__main__':
    main()
