import sys
import os 
import argparse
import numpy as np
import config
from experiments.model_representation import ModelRepresentationExtractor

def main(): 
    # create parser object    
    parser = argparse.ArgumentParser( \
        description='Extract model representation.', \
        epilog='By Yota Kawashima')
    
    # arguments
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
    parser.add_argument('caption_file_name', \
                        type=str, \
                        help='Caption file name.')
    
    # add default arguments
    parser.add_argument('--num_sentences_per_batch', \
                        type=int, \
                        default=config.num_sentences_per_batch, \
                        help='Number of sentences per batch.')    
    parser.add_argument('--data_dir_path', \
                        type=str, \
                        default=config.data_dir_path, \
                        help='Directory path for data.')
    parser.add_argument('--caption_dir_name', \
                        type=str, \
                        default=config.caption_dir_name, \
                        help='Directory name for caption data.')
    parser.add_argument('--model_representation_dir_name', \
                        type=str, \
                        default=config.model_representation_dir_name, \
                        help='Directory name for model representation data.')
    parser.add_argument('--model_representation_file_name', \
                        type=str, \
                        default=config.model_representation_file_name, \
                        help='File name for model representation data.')

    args = parser.parse_args()

    # extract and save model representations
    model = ModelRepresentationExtractor(args.model_name, args.checkpoint,
                                         data_dir_path = args.data_dir_path,
                                         model_representation_dir_name = args.model_representation_dir_name, 
                                         model_representation_file_name = args.model_representation_file_name)

    # data directory for captions and model representations
    caption_data_dir_path = \
        os.path.join(model.data_dir_path,  args.caption_dir_name)
    model_representation_data_dir_path = \
        os.path.join(model.data_dir_path, model.model_representation_dir_name, 
                     model.model_name + '_' + model.checkpoint,
                     args.model_representation_type + '_summary_type_' + args.model_representation_summary)

    os.makedirs(model_representation_data_dir_path, exist_ok=True)

    # create the data path to the caption data and the representation data
    caption_data_path = \
        os.path.join(caption_data_dir_path, args.caption_file_name)
    model_representation_file_name = \
        args.caption_file_name.replace('captions', model.model_representation_file_name)
    model_representation_file_name = \
        model_representation_file_name.replace('.pkl', '.npy')
    model_representation_data_path = \
        os.path.join(model_representation_data_dir_path, model_representation_file_name)

    # extrace model representation for the given caption file
    model_representation_array = \
        model.extract_model_representation_single_file(caption_data_path, 
                                                       args.num_sentences_per_batch, 
                                                       args.model_representation_type) 
    # summarise the model representation
    model.summarise_model_representation(model_representation_array, 
                                         model_representation_data_path,
                                         args.model_representation_summary)

if __name__ == '__main__':
    main()