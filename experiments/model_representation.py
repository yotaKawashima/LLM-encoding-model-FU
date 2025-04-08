import os 
import numpy as np
import torch 
import config
from transformers import AutoTokenizer, AutoModel, AutoConfig #, AutoModelForCausalLM 

class ModelRepresentationExtractor:
    def __init__(self, 
                 model_name, 
                 checkpoint, 
                 data_dir_path: str = config.data_dir_path,
                 model_representation_dir_name: str = config.model_representation_dir_name, 
                 model_representation_file_name: str = config.model_representation_file_name):
        """
        Initializes the ModelRepresentationExtractor with a model and tokenizer.
        Args:
            model_name (str): The name of the model to use. (e.g.EleutherAI/pythia-14m)
            checkpoint (str): The checkpoint to load. (e.g. step143000)
            data_dir_path (str): The path to the data directory.
            model_representation_dir_name (str): The name of the model representation directory.
            model_representation_file_name (str): The name of the model representation file.
        """
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.data_dir_path = data_dir_path
        self.model_representation_dir_name = model_representation_dir_name
        self.model_representation_file_name = model_representation_file_name

        # Load the model and tokenizer
        self.config = AutoConfig.from_pretrained(model_name, revision=checkpoint)
        self.model = AutoModel.from_pretrained(model_name, revision=checkpoint)
        #self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=checkpoint)
        
        # Set EOS token padding to the tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = self.tokenizer.model_max_length

        
    def extract_model_representation_batch(self, 
                                           list_inputs_to_model: list,
                                           model_representation_type: str) -> np.ndarray:
        """
        Extracts the representation of the text using the model."
        Args:
            list_inputs_to_model (list): a list of inputs to the model (before tokenisation)
            model_representation_type (str): The type of representation to extract.
                'last_hidden_state': The last hidden state of the model.
        Returns:
            np.ndarray: The extracted representation.
        """
        # Tokenize the input text
        inputs = \
            self.tokenizer(list_inputs_to_model, return_tensors="pt", padding=True)

        # Check if the input length exceeds the maximum length of the model
        if inputs['input_ids'].shape[1] > self.max_length:
            raise ValueError("Input length exceeds the maximum length of the model.")
        
        # Get the model outputs
        outputs = self.model(**inputs)
        
        if model_representation_type == 'last_hidden_state':        
            # Get the index of the last non-padding token for each sentence
            attention_mask = inputs["attention_mask"]
            last_token_indices = attention_mask.sum(dim=1) - 1  # shape: (batch_size,)

            # Use torch advanced indexing to get hidden states at the last real token
            # outputs.last_hidden_state: shape (batch_size, sequence_length, hidden_state_dim)
            batch_size = outputs.last_hidden_state.size(0)
            model_representation_last_token = \
                outputs.last_hidden_state[torch.arange(batch_size), last_token_indices]  # shape: (batch_size, hidden_dim)

            # conver to numpy array
            model_representation_last_token = model_representation_last_token.detach().numpy()
        else:
            raise ValueError("Invalid representation type. Choose implemented types.")
        
        return model_representation_last_token
    
    def extract_model_representation(self, 
                                     input_to_model: str,
                                     model_representation_type: str) -> np.ndarray:
        """
        Extracts the representation of the text using the model."
        Args:
            input_to_model (str): The input to the model (before tokenisation)
            model_representation_type (str): The type of representation to extract.
                'last_hidden_state': The last hidden state of the model.
        Returns:
            np.ndarray: The extracted representation.
        """
        # Tokenize the input text
        input = self.tokenizer(input_to_model, return_tensors="pt")
        if input['input_ids'].shape[1] > self.max_length:
            raise ValueError("Input length exceeds the maximum length of the model.")
        
        # Get the model outputs
        output = self.model(**input)
        
        if model_representation_type == 'last_hidden_state':
            # Get the last vector in the last hidden state of the model
            model_representation_last_token = output.last_hidden_state[0][-1]
        else: 
            raise ValueError("Invalid representation type. Choose implemented types.")
                
        return np.squeeze(model_representation_last_token.detach().numpy())
    
    def extract_model_representation_single_file(self, 
                                                 caption_data_path: str,
                                                 num_sentences_per_batch: int = config.num_sentences_per_batch,
                                                 model_representation_type: str = config.model_representation_type) -> np.ndarray:
        """
        Extracts the representation of sentences (in a single file) using the model. 
        Args:
            caption_data_path (str): The path to the captions data file.
            num_sentences_per_batch (int): The number of sentences to process in a batch.
            model_representation_type (str): The type of representation to extract.
                'last_hidden_state': The last hidden state of the model.
        Returns:
            np.ndarray: The extracted representation (num_image, num_sentences_per_image, hidden_dim).
        """
        # Load the captions data
        # captions data is a list of lists.
        # each list contains the captions for one image.
        # you may have multiple captions per image.
        caption_data = \
            np.load(caption_data_path, allow_pickle=True)
        
        # extract LLM representations for all captions.                 
        # Provide all sentences to the model as one batch and take average over the captions per image.
        # Get the number of sentences per image and the number of images
        num_sentences_per_images = len(caption_data[0])
        num_images = len(caption_data)
        num_sentences = num_sentences_per_images * num_images
        
        # Flatten the list of lists into a single list of sentences
        flattened_sentences = [sentence for sublist in caption_data for sentence in sublist]

        # Create subsets of the flattened sentences and process them in batches
        num_subsets = int(np.ceil(num_sentences / num_sentences_per_batch))
        
        # Initialize an empty list to store the model representations
        model_representation_array = []
        
        # Loop through the subsets
        for i_subset in range(num_subsets):
            # Get the start and end indices for the current subset
            start_index = i_subset * num_sentences_per_batch
            end_index = min((i_subset + 1) * num_sentences_per_batch, num_sentences)

            # Get the current subset of sentences
            subset_sentences = flattened_sentences[start_index:end_index]

            # Extract model representations for the current subset
            # model_representation_subset: shape (num_sentences_per_batch, hidden_dim)
            model_representation_subset = \
                self.extract_model_representation_batch(subset_sentences, 
                                                        model_representation_type)
            
            # Initialise model_representation_array
            if i_subset == 0:
                hidden_dim = model_representation_subset.shape[1]
                model_representation_array = \
                    np.zeros((num_sentences, hidden_dim))
            
            # store data
            model_representation_array[start_index:end_index, :] = \
                model_representation_subset

        # Reshape the array to (num_image, num_sentences_per_image, hidden_dim)
        model_representation_array = \
            model_representation_array.reshape(num_images, num_sentences_per_images, hidden_dim)

        return model_representation_array
    
    def summarise_model_representation(self, 
                                       model_representation_array: np.ndarray, 
                                       model_representation_data_path: str,
                                       model_representation_summary: str = config.model_representation_summary):
        """
        Summarises the model representation array.
        Args:
            model_representation_array (np.ndarray): The model representation array.
            model_representation_data_path (str): The path to save the summarised representation.
            model_representation_summary (str): how to summarise representation if you have multiple captions per image.
        Returns:
            np.ndarray: The summarised representation.
        """

        if model_representation_summary == 'mean': # mean representations across all captions per image
            # compute the mean along the 2nd axis (axis=1)
            model_representation_array_summarised = model_representation_array.mean(axis=1).squeeze()
        elif model_representation_summary == 'each': # keep all representations without summarisation.
            model_representation_array_summarised = model_representation_array 
        else: 
            raise ValueError("Invalid model_representation_summary. Choose implemented types.")

        # save the summarised representation 
        np.save(model_representation_data_path, model_representation_array_summarised)
        return

    def extract_and_summarise_model_representation_all_files(self,
                                                             caption_dir_name: str = config.caption_dir_name,
                                                             num_sentences_per_batch: int = config.num_sentences_per_batch,
                                                             model_representation_type: str = config.model_representation_type,
                                                             model_representation_summary: str = config.model_representation_summary):
        """
        Extracts the representation of a single sentence using the model."
        Args:
            caption_dir_name (str): The path to the captions data directory.
            num_sentences_per_batch (int): The number of sentences to process in a batch.
            model_representation_type (str): The type of representation to extract.
                'last_hidden_state': The last hidden state of the model.
            model_representation_summary (str): how to summarise representation if you have multiple captions per image.
                'mean': mean representations across all captions per image
                'each': keep all representations without summarisation.

        Returns:
            np.ndarray: The extracted representation.
        """
        # data directory for captions and model representations
        caption_data_dir_path = \
            os.path.join(self.data_dir_path,  caption_dir_name)
        model_representation_data_dir_path = \
            os.path.join(self.data_dir_path, self.model_representation_dir_name, 
                         self.model_name + '_' + self.checkpoint, 
                         model_representation_type + '_summary_type_' + model_representation_summary)
        
        os.makedirs(model_representation_data_dir_path, exist_ok=True)

        # list all file name in the captions data directory
        caption_file_names = \
            [f for f in os.listdir(caption_data_dir_path) if os.path.isfile(os.path.join(caption_data_dir_path, f))]
        
        # extract model representaion for each caption
        for caption_file_name in caption_file_names:
            # create the data path to the caption data and the representation data
            caption_data_path = \
                os.path.join(caption_data_dir_path, caption_file_name)
            model_representation_file_name = \
                caption_file_name.replace('captions', self.model_representation_file_name)
            model_representation_file_name = \
                model_representation_file_name.replace('.pkl', '.npy')
            model_representation_data_path = \
                os.path.join(model_representation_data_dir_path, model_representation_file_name)

            model_representation_array = \
                self.extract_model_representation_single_file(caption_data_path, 
                                                              num_sentences_per_batch, 
                                                              model_representation_type) 
            # summarise the model representation
            self.summarise_model_representation(model_representation_array, 
                                                model_representation_data_path,
                                                model_representation_summary)
                    
        return 