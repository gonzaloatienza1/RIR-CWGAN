""" 
This script evaluates multiple RIR models saved during training and selects the best model based on the NMSE (Normalized Mean Square Error) metric.
Models are saved every 5 epochs during training, and this script iteratively loads them, calculates NMSE, and saves the results.
"""

import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from model.cwgan_model import Generator
from ..misc.load_data import load_data
from ..misc.align_rirs import align_rirs
from metrics.nmse_db_loss import NMSEdBLoss
from dataloader.rir_dataset import RIRDataset
import argparse

def search_model(output_path: str, cache_rir: str, cache_emb: str, num_epochs: int = 500, z_dim: int = 50, batch_size: int = 1, target_samples: int = 6000) -> None:
    """
    Evaluate multiple CWGAN generator checkpoints by computing NMSE over test RIRs.

    Args:
        output_path (str): Directory containing saved generator models (e.g., "generator_epch5.pth").

        cache_rir (str): Path to the cached RIRs pickle file.

        cache_emb (str): Path to the cached embeddings pickle file.

        num_epochs (int, optional): Maximum number of epochs to evaluate. Defaults to 500.

        z_dim (int, optional): Dimensionality of the noise vector used by the generator. Defaults to 50.

        batch_size (int, optional): Batch size for evaluation. Defaults to 1.

        target_samples (int, optional): Number of samples to retain in aligned RIRs. Defaults to 6000.

    Output:
        - Saves an Excel file `NMSE_summary.xlsx` in `output_path` with NMSE mean and variance per epoch.
    """
    # Load test data
    _, emb_test, _, rirs_test, _, _, _, _, _, _ = load_data(cache_rir, cache_emb)
    test_dataset = RIRDataset(rirs=rirs_test, embeddings=emb_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nmse_data = {
        'epoch': [],
        'NMSE 6000 mean': [],
        'NMSE 6000 variance': []
    }

    for epoch in range(5, num_epochs + 1, 5):
        model_path = os.path.join(output_path, f"generator_epch{epoch}.pth")

        if os.path.exists(model_path):
            print(f"Processing model at epoch {epoch}")

            generator = Generator(z_dim)
            generator.load_state_dict(torch.load(model_path, map_location=device))
            generator.to(device)

            nmse_full = calculate_nmse(generator, test_loader, z_dim, device, target_samples)
            nmse_data['epoch'].append(epoch)
            nmse_data['NMSE 6000 mean'].append(np.mean(nmse_full))
            nmse_data['NMSE 6000 variance'].append(np.var(nmse_full))

        else:
            print(f"Model for epoch {epoch} not found.")

    save_nmse_summary_to_excel(nmse_data, output_path)


def calculate_nmse(generator, dataloader, z_dim, device, target_samples=6000):
    """
    Compute NMSE (in decibels) between generated and ground truth RIRs over the full dataset.

    Args:
        generator (nn.Module): Trained generator model.
        dataloader (DataLoader): DataLoader providing batches of real RIRs and embeddings.
        z_dim (int): Dimension of the latent noise vector.
        device (torch.device): Computation device (CPU or GPU).
        target_samples (int): Number of samples to retain in aligned RIRs (defaults to 6000).

    Returns:
        List[float]: List of NMSE (in dB) values, one per test RIR.
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
    Save a summary of NMSE statistics (mean and variance) per epoch to an Excel file.

    Args:
        nmse_data (dict): Dictionary with 'epoch', 'NMSE 6000 mean', 'NMSE 6000 variance'.
        output_path (str): Directory path to save 'NMSE_summary.xlsx'.
    """
    # Create a DataFrame and save it to the specified path
    df = pd.DataFrame(nmse_data)
    excel_file_path = os.path.join(output_path, 'NMSE_summary.xlsx')
    df.to_excel(excel_file_path, index=False)
    print(f"NMSE summary saved to: {excel_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple CWGAN generator models based on NMSE.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory containing saved generator models.")
    parser.add_argument('--cache_rir', type=str, required=True, help="Path to cached RIRs pickle file.")
    parser.add_argument('--cache_emb', type=str, required=True, help="Path to cached embeddings pickle file.")
    parser.add_argument('--num_epochs', type=int, default=500, help="Max number of training epochs to evaluate.")
    args = parser.parse_args()

    search_model(
        output_path=args.output_path,
        cache_rir=args.cache_rir,
        cache_emb=args.cache_emb,
        num_epochs=args.num_epochs
    )
