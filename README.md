# **RIR-CWGAN: Conditional Wasserstein Generative Adversarial Network for Synthesizing Realistic Room Impulse Responses**

This repository provides the implementation of a **CWGAN-based** approach for Room Impulse Responses (RIRs) generation using embeddings that encapsulate room properties. The method is designed to model acoustic responses across various room conditions, employing neural networks to generate RIRs that closely match measured or simulated responses. Predicting RIRs is crucial for audio scene modeling, speech enhancement, and room acoustics simulation.

Key features of this repository include:

- **Custom RIR datasets:** Handling real and simulated RIRs using embeddings.

- **Generator and Critic networks:** Trained using a Wasserstein GAN approach.

- **Evaluation metrics:** Quantifying prediction accuracy using:

    - Normalized Mean Squared Error (NMSE)

    - Normalized Projection Misalignment (NPM)

    - Direct-to-Reverberant Ratio Difference (DRR)

    - Early-to-Total Sound Energy Ratio Difference (D50)

## **Requirements**

**Python Version:** 3.10.12

**Dependencies:**

```
librosa==0.9.2
matplotlib==3.8.0
numpy==1.24.4
pandas==1.5.3
scikit_learn==1.2.0
scipy==1.15.1
torch==2.1.0
tqdm==4.66.1
```

## **Usage**

### **Embedding Processing** (`embedding.py`)

The `embedding.py` script processes and normalizes room configuration embeddings derived from audio file metadata. These embeddings encapsulate room properties such as dimensions, listener and speaker positions, reverberation times (T60), and distances. The processed embeddings are saved in a pickle file for subsequent use in RIR generation models.

Use this script before training to preprocess room data and generate embeddings required by the model.

```
python src/embedding.py --data_folder path/to/audio_data --cache_file path/to/cache.pkl --output_embedding_file path/to/normalized_embeddings.pkl
```

**Parameters:**

- `--data_folder`: Directory containing the audio data.

- `--cache_file`: Path to store/load the cache of audio data.

- `--output_embedding_file`: Path to save the normalized embeddings.

### **CWGAN Training** (`trainCWGAN.py`)

The `trainCWGAN.py` script is responsible for training a Conditional Wasserstein GAN (CWGAN) to generate RIRs. The training alternates between optimizing the generator and the critic networks using the Wasserstein loss. During the training process, the generator tries to produce RIRs that are indistinguishable from real ones, while the critic learns to differentiate between real and generated RIRs.

Use this script to train the model on room impulse response data after preparing the necessary embeddings and datasets.

```
python src/trainCWGAN.py --cache_rir path/to/cached_rir.pkl --cache_emb path/to/cached_embeddings.pkl --num_epochs 500 --batch_size 256 --z_dim 50 --gen_lr 5e-5 --crit_lr 5e-5
```

**Parameters:**

- `--cache_rir`: Path to the cached RIR data.

- `--cache_emb`: Path to the cached embeddings.

- `--num_epochs`: Number of epochs for training (default: 500).

- `--batch_size`: Batch size for training (default: 256).

- `--z_dim`: Dimensionality of the noise vector (default: 50).

- `--gen_lr`: Learning rate for the generator (default: 5e-5).

- `--crit_lr`: Learning rate for the critic (default: 5e-5).

### **Model Evaluation** (`search_model.py`)

The `search_model.py` script evaluates different RIR models saved during the training process and selects the best one based on the Normalized Mean Square Error (NMSE) metric. During training, models are saved every 5 epochs, and this script iteratively loads each saved model, calculates NMSE, and saves the results to an Excel file.

This script is used post-training to evaluate and compare the performance of saved models and select the best-performing model based on the NMSE metric.

```
python src/search_model.py --output_path path/to/output --cache_rir path/to/cached_rir.pkl --cache_emb path/to/cached_embeddings.pkl --num_epochs 500
```

**Parameters:**

- `--output_path`: Directory where the NMSE summary will be saved.

- `--cache_rir`: Path to the cached RIR data.

- `--cache_emb`: Path to the cached embeddings.

- `--num_epochs`: Number of epochs for which models were saved (default: 500).

### **CWGAN Evaluation** (`testCWGAN.py`)

The `testCWGAN.py` script evaluates a trained CWGAN generator. It performs two main functions:

1. Compute and save error metrics comparing generated and real RIRs to an Excel file.

2. Plot and save 100 randomly selected RIR comparisons to visualize the generator's performance.

Use this script to assess the performance of a trained CWGAN model by comparing the generated RIRs with the real ones and evaluating various error metrics.

```
python src/testCWGAN.py --model_path path/to/generator.pth --output_path path/to/output --cache_rir path/to/cached_rir.pkl --cache_emb path/to/cached_embeddings.pkl --generate_excel --plot_rirs
```

**Parameters:**

- `--model_path`: Path to the trained generator model file.

- `--output_path`: Directory where outputs (plots and/or Excel files) will be saved.

- `--cache_rir`: Path to the cached RIR pickle file.

- `--cache_emb:` Path to the cached embeddings pickle file.

- `--generate_excel:` If provided, calculates and saves error metrics to an Excel file.

- `--plot_rirs`: If provided, plots and saves 100 randomly selected RIR comparisons.

## **Citation**

This work is currently under submission. Once published, the citation details will be updated here.