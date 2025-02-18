""" 
This script evaluates multiple RIR models saved during training and selects the best model based on the NMSE (Normalized Mean Square Error) metric.
Models are saved every 5 epochs during training, and this script iteratively loads them, calculates NMSE, and saves the results.
"""
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from model import Generator
from utils import load_data, align_rirs
from metrics import NMSEdBLoss
from rir_dataset import RIRDataset
import argparse

# Set device to GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_nmse(generator, dataloader, z_dim, device, target_samples=6000):
    """
    Calculate the NMSE for each pair of real and generated RIRs.
    generator: Trained model to generate RIRs.
    dataloader: DataLoader providing batches of RIRs and embeddings.
    z_dim: Dimensionality of the noise vector.
    device: Device to run the calculations on (CPU/GPU).
    target_samples: Number of samples to consider in NMSE calculation.
    """
    criterion_nmse = NMSEdBLoss()
    generator.eval()
    nmse_full = []

    with torch.no_grad():
        for rirs, embeddings in dataloader:
            # Move data to the specified device
            rirs, embeddings = rirs.to(device), embeddings.to(device)
            # Generate noise input
            z_noise = torch.randn(embeddings.size(0), z_dim).to(device)
            # Generate RIRs using the generator
            generated_rirs = generator(embeddings, z_noise)
            # Convert tensors to NumPy arrays for alignment
            rirs = rirs.squeeze().cpu().numpy()
            generated_rirs = generated_rirs.squeeze().cpu().numpy()

            # Align the real and generated RIRs
            aligned_real_rir, aligned_generated_rir, _ = align_rirs(rirs, generated_rirs)
            # Convert aligned RIRs back to tensors
            aligned_real_rir = torch.from_numpy(aligned_real_rir[:target_samples]).to(device)
            aligned_generated_rir = torch.from_numpy(aligned_generated_rir[:target_samples]).to(device)

            # Calculate NMSE for the aligned signals
            nmse_loss_full = criterion_nmse(aligned_generated_rir, aligned_real_rir).item()
            nmse_full.append(nmse_loss_full)

    return nmse_full

def save_nmse_summary_to_excel(nmse_data, output_path):
    """
    Save the NMSE results to an Excel file.
    nmse_data: Dictionary containing NMSE values for each epoch.
    output_path: Directory to save the output Excel file.
    """
    # Create a DataFrame and save it to the specified path
    df = pd.DataFrame(nmse_data)
    excel_file_path = os.path.join(output_path, 'NMSE_summary.xlsx')
    df.to_excel(excel_file_path, index=False)
    print(f"NMSE summary saved to: {excel_file_path}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Script to calculate NMSE for RIR models")
    parser.add_argument('--output_path', type=str, required=True, help="Output path to save the NMSE summary")
    parser.add_argument('--cache_rir', type=str, required=True, help="Path to cache_rir.pkl")
    parser.add_argument('--cache_emb', type=str, required=True, help="Path to cache_emb.pkl")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs to train.")
    args = parser.parse_args()

    # Set model and data parameters
    z_dim = 50
    batch_size = 1

    # Load data from cached pickle files
    emb_train, emb_test, rirs_train, rirs_test, distance_train, distance_test, file_names_train, file_names_test = load_data(args.cache_rir, args.cache_emb)

    # Create a DataLoader for the test set
    test_dataset = RIRDataset(rirs=rirs_test, embeddings=emb_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the dictionary to store NMSE results
    nmse_data = {
        'epoch': [],
        'NMSE 6000 mean': [],
        'NMSE 6000 variance': []
    }

    # Iterate through saved models and calculate NMSE
    for epoch in range(5, args.num_epochs + 1, 5):
        model_path = os.path.join(args.output_path, f"generator_epch{epoch}.pth")

        if os.path.exists(model_path):
            print(f"Processing model at epoch {epoch}")

            # Load the saved generator model
            generator = Generator(z_dim)
            generator.load_state_dict(torch.load(model_path, map_location=device))
            generator.to(device)

            # Calculate NMSE for the current model
            nmse_full = calculate_nmse(generator, test_loader, z_dim, device, target_samples=6000)
            nmse_full_mean = np.mean(nmse_full)
            nmse_full_var = np.var(nmse_full)

            # Store results
            nmse_data['epoch'].append(epoch)
            nmse_data['NMSE 6000 mean'].append(nmse_full_mean)
            nmse_data['NMSE 6000 variance'].append(nmse_full_var)

        else:
            print(f"Model for epoch {epoch} not found.")

    # Save NMSE summary to an Excel file
    save_nmse_summary_to_excel(nmse_data, args.output_path)

