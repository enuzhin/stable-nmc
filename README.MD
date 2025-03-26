# Stability-aware Neuromorphic Computing

This project includes training on the MNIST dataset. The neural network was trained with noisy weights constrained to positive values within a limited range. Nevertheless, it achieved reliable accuracy.

This repository contains a machine learning project that represents the results of the paper.

## Project Structure

```
/project_files
│── eval.py                # Script for evaluating the model
│── train.py               # Script for training the model
│── utils.py               # Utility functions used across the project
│── requirements.txt       # Dependencies list
│── config/                # Configuration files for the model
│   ├── config.yaml        # Main configuration file
│   ├── model/             
│   │   ├── res_net.yaml   # Configuration for ResNet model
│   │   ├── feed_forward.yaml # Configuration for Feed Forward model
│── models/                #
│   ├── noisy_archs.py     # Contains model architectures with noise handling
│   ├── noisy_base.py      # Base module for working with noisy models
│   ├── __init__.py        # Initialization file for module usage
│── checkpoints/           # Stores trained model checkpoints
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/enuzhin/stable-nmc.git
   cd stable-nmc
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Model configurations are stored in `config/model/`. You can specify the model type when running the scripts.

## Training

Run the following command to train a model:
   ```bash
   python train.py model.<model_type>
   ```
where `<model_type>` is `res_net` or `feed_forward`.

This will train the model using the corresponding configuration.

## Evaluation

Run the following command to evaluate a trained model:
   ```bash
   python eval.py model.<model_type>
   ```
where `<model_type>` is `res_net` or `feed_forward`.

## Publication

The related publication will be available soon.

