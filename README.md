# **RIR-CWGAN: Conditional Wasserstein Generative Adversarial Network for Synthesizing Realistic Room Impulse Responses**

This repository provides the implementation of a **CWGAN-based** approach for Room Impulse Responses (RIRs) generation using embeddings that encapsulate room properties. The method is designed to model acoustic responses across various room conditions, employing neural networks to generate RIRs that closely match measured or simulated responses. Predicting RIRs is crucial for audio scene modeling, speech enhancement, and room acoustics simulation.

Key features of this repository include:

- **Custom RIR datasets:** Handling real and simulated RIRs using embeddings.
- **Generator and Critic networks:** Trained using a Wasserstein GAN approach.
- **Evaluation metrics:** Quantifying prediction accuracy using:
    - Normalized Mean Squared Error (NMSE)
    - Normalized Projection Misalignment (NPM)
    - Direct-to-Reverberant Ratio Difference (ΔDRR)
    - Early-to-Total Sound Energy Ratio Difference (ΔD50)

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
The script processes and normalizes room configuration embeddings derived from audio file metadata. These embeddings encapsulate room properties such as dimensions, listener and speaker positions and reverberation times (T60). The processed embeddings are saved in a pickle file for subsequent use in RIR generation models.

Use this script before training to preprocess room data and generate embeddings required by the model.
```
python src/misc/embedding.py --data_folder path/to/audio_data --cache_file path/to/cache.pkl --output_embedding_file path/to/normalized_embeddings.pkl
```
**Parameters:**
- `--data_folder`: Directory containing the audio data.
- `--cache_file`: Path to store/load the cache of audio data.
- `--output_embedding_file`: Path to save the normalized embeddings.

### **CWGAN Training** (`cwgan_train.py`)
Trains a Conditional Wasserstein GAN (CWGAN) to generate RIRs. The training process alternates between optimizing the generator and critic using the Wasserstein loss, saving models every 5 epochs.
```
python src/train/cwgan_train.py --cache_rir path/to/cached_rir.pkl --cache_emb path/to/cached_embeddings.pkl --num_epochs 500 --batch_size 256 --z_dim 50 --gen_lr 5e-5 --crit_lr 5e-5
```
**Parameters:**
- `--cache_rir`: Path to cached RIR data.
- `--cache_emb`: Path to cached embeddings.
- `--num_epochs`: Number of training epochs (default: 500).
- `--batch_size`: Batch size (default: 256).
- `--z_dim`: Latent noise dimension (default: 50).
- `--gen_lr`: Generator learning rate (default: 5e-5).
- `--crit_lr`: Critic learning rate (default: 5e-5).

### **Model Evaluation** (`search_model.py`)
Evaluates CWGAN models saved during training. Iterates through saved models (every 5 epochs), calculates NMSE, and saves a summary to Excel.
```
python src/test/search_model.py --output_path path/to/output --cache_rir path/to/cached_rir.pkl --cache_emb path/to/cached_embeddings.pkl --num_epochs 500
```
**Parameters:**
- `--output_path`: Directory to save the Excel summary.
- `--cache_rir`: Path to cached RIR data.
- `--cache_emb`: Path to cached embeddings.
- `--num_epochs`: Number of saved epochs (default: 500).

### **CWGAN Metric Evaluation** (`cwgan_test.py`)
Evaluates a single trained CWGAN model. It calculates error metrics (MSE, NMSE, NPM, ΔDRR, ΔD50) and saves them to Excel.
```
python src/test/cwgan_test.py --model_path path/to/generator.pth --output_path path/to/output --cache_rir path/to/cached_rir.pkl --cache_emb path/to/cached_embeddings.pkl --z_dim 50 --batch_size 1
```
**Parameters:**
- `--model_path`: Path to the trained generator model.
- `--output_path`: Output directory to save Excel file.
- `--cache_rir`: Path to cached RIR data.
- `--cache_emb`: Path to cached embeddings.
- `--z_dim`: Noise vector dimension (default: 50).
- `--batch_size`: Batch size (default: 1).

### **CWGAN RIR Visualization** (`cwgan_viz.py`)
Generates and plots RIRs from a trained model. Compares generated vs real RIRs visually and annotates with the embedding.
```
python src/viz/cwgan_viz.py --model_path path/to/generator.pth --output_path path/to/output --cache_rir path/to/cached_rir.pkl --cache_emb path/to/cached_embeddings.pkl --z_dim 50 --n_plots 100
```
**Parameters:**
- `--model_path`: Path to the trained generator model.
- `--output_path`: Directory to save the plots.
- `--cache_rir`: Path to cached RIR data.
- `--cache_emb`: Path to cached embeddings.
- `--z_dim`: Noise vector dimension (default: 50).
- `--n_plots`: Number of RIR plots to generate (default: 100).

## **Citation**
This work is currently under submission. Once published, the citation details will be updated here.