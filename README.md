# LLM-encoding-model-FU

## Overview
This project explores natural scene representations in the brain through large language models (LLMs). 
For a detailed description of the project goals, methodology, and expected outcomes, please refer to the [project proposal](./proposal.pdf).

## Features
- **Model Representations**: Extraction and normalization of LLM representations.
- **Encoding Models**: Implementation of encoding models using Ridge regression and cross-validation.
- **Task Evaluation**: Integration with `lm-evaluation-harness` for testing LLMs on various tasks.

## Getting Started

### Prerequisites
- Python 3.8 or higher

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yotaKawashima/LLM-encoding-model-FU.git
    cd LLM-encoding-model-FU
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-lm-eval.txt  # For lm-evaluation-harness
    ```

### Usage


## Project Structure
- `experiments/`: Scripts for extracting representations, running encoding models, and data visualisation.
- `data/`: Directory for storing data (captions, fMRI, etc). You need to add data in this directory. 

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [EleutherAI](https://www.eleuther.ai/) for the `lm-evaluation-harness` and open-access LLMs.
- [Hugging Face](https://huggingface.co/) for the `transformers` library.
- [scikit-learn](https://scikit-learn.org/) for machine learning tools.
- [PyTorch](https://pytorch.org/) for deep learning frameworks.