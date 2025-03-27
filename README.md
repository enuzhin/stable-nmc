# Stability-aware Neuromorphic Computing

This repository presents a model trained on the MNIST dataset. The neural network was trained under extreme noise condition and weights constrained to positive values within a limited range. Nevertheless, it achieved reliable accuracy.

The model is inspired by the inherent noise observed in memristive devices, which have the potential to serve as power-efficient physical representations of neural network weights in the neuromorphic computing paradigm.

The table below summarizes the performance of different models trained with and without hardware-inspired weights constraints:

| Model             | Percentage Error (%) | Accuracy (%) | Trainable Parameters | Cross-Entropy Loss |
|------------------|----------------------|--------------|----------------------|---------------------|
| NoisyResNet       | 0.33                 | 99.67        | 8,008,980            | 0.00051             |
| ResNet            | 0.31                 | 99.69        | 3,662,100            | 0.00051             |
| NoisyFeedForward  | 1.10                 | 98.90        | 8,008,980            | 0.00054             |
| FeedForward       | 0.62                 | 99.38        | 3,662,100            | 0.00052             |


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



