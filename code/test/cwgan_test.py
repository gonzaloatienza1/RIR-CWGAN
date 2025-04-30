"""
This script evaluates a trained CWGAN generator by calculating various error metrics
between generated and real Room Impulse Responses (RIRs) and exporting the results to Excel.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from typing import List, Tuple
from models.cwgan_model import Generator
from metrics.nmse_db_loss import NMSEdBLoss
from metrics.npm import  NPM
from metrics.d50 import  D50
from metrics.drr import DRR
from metrics.rmse import RMSE
from dataloader.rir_dataset import RIRDataset
import argparse
from ..misc.load_data import load_data
from ..misc.align_rirs import align_rirs

def calculate_errors(
    generator: nn.Module,
    dataloader: DataLoader,
    z_dim: int,
    device: torch.device
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Evaluate the generator by comparing generated and ground-truth RIRs using multiple metrics.

    Args:
        generator (nn.Module): Trained CWGAN Generator.

        dataloader (DataLoader): PyTorch DataLoader providing batches of real RIRs and embeddings.

        z_dim (int): Dimensionality of the latent noise vector input to the generator.
        
        device (torch.device): Computation device (CPU or GPU).

    Returns:
        Tuple[List[float], List[float], List[float], List[float], List[float]]: A tuple containing:

            - **mse_losses** (List[float]): Per-sample Mean Squared Error values.
            - **nmse_losses** (List[float]): Per-sample Normalized MSE in decibels.
            - **npm_losses** (List[float]): Per-sample Normalized Projection Misalignment scores.
            - **drr_losses** (List[float]): Per-sample DRR error (RMSE between predicted and true DRR).
            - **D50_losses** (List[float]): Per-sample D50 error (RMSE between predicted and true D50).
    """
    criterion1 = nn.MSELoss()
    criterion2 = NMSEdBLoss()
    criterion4 = NPM()
    criterion5 = DRR()
    criterion6 = D50()
    rmse = RMSE()

    test_losses_mse, test_losses_nmse, test_losses_npm = [], [], []
    test_losses_drr, test_losses_D50 = [], []

    generator.eval()
    with torch.no_grad():
        # Generate RIRs using the trained generator
        # Iterate through the DataLoader to process each batch of RIRs and embeddings
        for rirs, embeddings in dataloader:
            rirs, embeddings = rirs.to(device), embeddings.to(device)
            z_noise = torch.randn(embeddings.size(0), z_dim).to(device)
            generated_rirs = generator(embeddings, z_noise)

            rirs = rirs.squeeze().cpu().numpy()
            generated_rirs = generated_rirs.squeeze().cpu().numpy()

            # Align the real and generated RIRs for comparison
            adj_real_rir, adj_generated_rir, _ = align_rirs(rirs, generated_rirs)
            adj_real_rir = torch.from_numpy(adj_real_rir).to(device)
            adj_generated_rir = torch.from_numpy(adj_generated_rir).to(device)

            mse_loss = criterion1(adj_generated_rir, adj_real_rir).item()
            nmse_loss = criterion2(adj_generated_rir, adj_real_rir).item()
            npm_loss = criterion4(adj_generated_rir, adj_real_rir).item()

            n1 = 350
            drr_pred = criterion5(adj_generated_rir, nd=n1).item()
            drr_true = criterion5(adj_real_rir, nd=n1).item()
            loss_drr_diff = rmse(drr_true, drr_pred)

            D50_pred = criterion6(adj_generated_rir, ne=n1).item()
            D50_true = criterion6(adj_real_rir, ne=n1).item()
            loss_D50_diff = rmse(D50_true, D50_pred)

            test_losses_mse.append(mse_loss)
            test_losses_nmse.append(nmse_loss)
            test_losses_npm.append(npm_loss)
            test_losses_drr.append(loss_drr_diff)
            test_losses_D50.append(loss_D50_diff)

    # Return the calculated metrics for all batches
    return test_losses_mse, test_losses_nmse, test_losses_npm, test_losses_drr, test_losses_D50

def save_errors_to_excel(
    rirs_filenames: List[str],
    distances: List[float],
    mse_losses: List[float],
    nmse_losses: List[float],
    npm_losses: List[float],
    drr_losses: List[float],
    D50_losses: List[float],
    output_path: str
) -> None:
    """
    Save per-sample error metrics and global statistics (mean, variance) to an Excel file.

    Args:
        rirs_filenames (List[str]): Filenames associated with each test RIR.
        distances (List[float]): Euclidean distances between source and microphone.
        mse_losses (List[float]): MSE losses for each RIR.
        nmse_losses (List[float]): NMSE losses for each RIR.
        npm_losses (List[float]): NPM scores for each RIR.
        drr_losses (List[float]): DRR error (RMSE) for each RIR.
        D50_losses (List[float]): D50 error (RMSE) for each RIR.
        output_path (str): Output directory where 'error_metrics.xlsx' will be saved.

    Returns:
        None
    """
    # Prepare the data dictionary with metrics and their statistics
    data = {
        'RIR File': rirs_filenames + ['MEAN', 'VARIANCE'],
        'MSE Loss': mse_losses + [np.mean(mse_losses), np.var(mse_losses)],
        'NMSE Loss': nmse_losses + [np.mean(nmse_losses), np.var(nmse_losses)],
        'NormalizedProjectionMisalignment': npm_losses + [np.mean(npm_losses), np.var(npm_losses)],
        'DirectToReverberantRatio_diff': drr_losses + [np.nanmean(drr_losses), np.nanvar(drr_losses)],
        'EarlyToTotalSoundEnergyRatio_diff': D50_losses + [np.mean(D50_losses), np.var(D50_losses)],
        'Distance': distances + ['', '']
    }
    # Save the DataFrame to an Excel file
    df = pd.DataFrame(data)
    excel_file_path = os.path.join(output_path, 'error_metrics.xlsx')
    df.to_excel(excel_file_path, index=False)
    print(f"Error metrics saved to: {excel_file_path}")    

if __name__ == "__main__":
    # Parse command-line arguments for the evaluation script
    parser = argparse.ArgumentParser(description="Evaluate a trained CWGAN model on test data and save error metrics.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the generator model file.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save Excel output.")
    parser.add_argument('--cache_rir', type=str, required=True, help="Path to the cached RIR pickle file.")
    parser.add_argument('--cache_emb', type=str, required=True, help="Path to the cached embeddings pickle file.")
    parser.add_argument('--z_dim', type=int, default=50, help="Dimensionality of the noise vector.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for test DataLoader.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the generator model and load its trained weights
    generator = Generator(z_dim=args.z_dim)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    generator.to(device)

    # Load the test data and embeddings from cached files
    emb_train, emb_test, rirs_train, rirs_test, distance_train, distance_test, file_names_train, file_names_test, _, _ = load_data(
        args.cache_rir, args.cache_emb)

    # Create the test dataset and DataLoader
    test_dataset = RIRDataset(rirs=rirs_test, embeddings=emb_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Evaluate the generator on the test dataset and calculate error metrics
    mse_losses, nmse_losses, npm_losses, drr_losses, D50_losses = calculate_errors(
        generator, test_loader, args.z_dim, device
    )

    # Save the calculated metrics to an Excel file
    save_errors_to_excel(
        file_names_test, distance_test,
        mse_losses, nmse_losses, npm_losses,
        drr_losses, D50_losses,
        args.output_path
    )
