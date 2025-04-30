"""
This script trains a Conditional Wasserstein Generative Adversarial Network (CWGAN)
for generating Room Impulse Responses (RIRs) with modular training steps.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cwgan_model import Generator, Critic
from dataloader.rir_dataset import RIRDataset
from ..misc.load_data import load_data
from ..misc.align_rirs import align_rirs
from ..misc.embedding import destandardize_data
from typing import Tuple
import os
import numpy as np
import time
import argparse

def forward_batch(generator: torch.nn.Module, critic: torch.nn.Module, rirs: torch.Tensor, embeddings: torch.Tensor, z_dim: int, opt_gen: torch.optim.Optimizer, opt_critic: torch.optim.Optimizer,
                  clip_value: float, critic_iterations: int, device: torch.device) -> Tuple[float, float]:
    """
    Performs a single batch update including multiple critic steps and one generator step.

    Args:
        generator (nn.Module): Generator model.

        critic (nn.Module): Critic model.

        rirs (torch.Tensor): A 3D tensor of shape (batch_size, 1, rir_length) that represents the real RIRs, where:
            
            - **batch_size**: The number of samples per batch.
            - **rir_length**: Number of samples per audio signal.

        embeddings (torch.Tensor): A 2D tensor of shape (batch_size, 10) that represents the conditioning embeddings.

        z_dim (int): Latent noise dimension.

        opt_gen (torch.optim.Optimizer): Optimizer for the generator.

        opt_critic (torch.optim.Optimizer): Optimizer for the critic.

        clip_value (float): Value for critic weight clipping.

        critic_iterations (int): Number of critic updates per batch.

    Returns:
        Tuple[float, float]: A tuple containing the following elements:

            - **float**: The loss value of the Generator model for the current batch.
            - **float**: The loss value of the Critic model for the current batch.          
    """
    rirs, embeddings = rirs.to(device), embeddings.to(device)
    batch_size = rirs.size(0)

    # Perform multiple critic updates for each batch
    for _ in range(critic_iterations):
        opt_critic.zero_grad()
        real_outputs = critic(rirs, embeddings)
        fake_rirs = generator(embeddings, torch.randn(batch_size, z_dim, device=device))
        fake_outputs = critic(fake_rirs.detach(), embeddings)
        d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs)
        d_loss.backward()
        opt_critic.step()
        for p in critic.parameters():
            p.data.clamp_(-clip_value, clip_value)

    opt_gen.zero_grad()
    fake_rirs = generator(embeddings, torch.randn(batch_size, z_dim, device=device))
    # Calculate the generator loss (maximize critic score for fake RIRs)
    g_loss = -torch.mean(critic(fake_rirs, embeddings))
    g_loss.backward()
    opt_gen.step()

    return g_loss.item(), d_loss.item()

def forward_epoch(generator: torch.nn.Module, critic: torch.nn.Module, loader: DataLoader, z_dim: int, opt_gen: torch.optim.Optimizer, opt_critic: torch.optim.Optimizer,
                  clip_value: float, critic_iterations: int, device: torch.device) -> Tuple[float, float]:
    """
    Trains one full epoch.

    Args:
        generator (nn.Module): Generator model.

        critic (nn.Module): Critic model.

        loader (DataLoader): DataLoader for the training set.

        z_dim (int): Latent noise dimension.

        opt_gen (torch.optim.Optimizer): Generator optimizer.

        opt_critic (torch.optim.Optimizer): Critic optimizer.

        clip_value (float): Weight clipping for critic.

        critic_iterations (int): Number of critic steps per batch.

    Returns:
        Tuple[float, float]: A tuple containing the following elements:

            - **float**: The average loss value (of all batches) of the Generator model for the current epoch.
            - **float**: The average loss value (of all batches) of the Critic model for the current epoch.     
    """
    gen_loss_log, critic_loss_log = [], []
    for rirs, embeddings in loader:
        g_loss, d_loss = forward_batch(generator, critic, rirs, embeddings, z_dim, opt_gen, opt_critic, clip_value, critic_iterations, device)
        gen_loss_log.append(g_loss)
        critic_loss_log.append(d_loss)
    
    # Return the average generator and critic losses for the epoch
    return sum(gen_loss_log)/len(gen_loss_log), sum(critic_loss_log)/len(critic_loss_log)

def train_model(generator: torch.nn.Module, critic: torch.nn.Module, loader: DataLoader, num_epochs: int, z_dim: int, gen_lr: float, crit_lr: float, device: torch.device, clip_value: float = 0.01, critic_iterations: int = 5) -> None:
    """
    Main training loop for CWGAN.

    Args:
        generator (nn.Module): Generator network.

        critic (nn.Module): Critic network.

        loader (DataLoader): Training data loader.

        num_epochs (int): Total number of epochs.

        z_dim (int): Latent dimension for noise input.

        gen_lr (float): Learning rate for generator.

        crit_lr (float): Learning rate for critic.

        clip_value (float, optional): Weight clipping value. Defaults to 0.01.

        critic_iterations (int, optional): Number of critic updates per batch. Defaults to 5.
    """
    opt_gen = optim.RMSprop(generator.parameters(), lr=gen_lr)
    opt_critic = optim.RMSprop(critic.parameters(), lr=crit_lr)

    scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=50, gamma=0.5)
    scheduler_critic = optim.lr_scheduler.StepLR(opt_critic, step_size=50, gamma=0.5)

    for epoch in range(num_epochs):
        start_time = time.time()
        g_loss, d_loss = forward_epoch(generator, critic, loader, z_dim, opt_gen, opt_critic, clip_value, critic_iterations, device)
        scheduler_gen.step()
        scheduler_critic.step()
        print(f"Epoch {epoch+1}/{num_epochs} - G_loss: {g_loss:.4f} - D_loss: {d_loss:.4f} - Time: {time.time() - start_time:.2f}s")

        if (epoch + 1) % 5 == 0:
            os.makedirs("CWGAN", exist_ok=True)
            torch.save(generator.state_dict(), f"CWGAN/generator_epch{epoch + 1}.pth")

def model_test(generator: torch.nn.Module, rir_real: np.ndarray, embedding: np.ndarray, means: np.ndarray, stds: np.ndarray, z_dim: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic RIR from a given embedding using a trained generator,
    align it with the corresponding real RIR, and de-standardize the embedding.

    Args:
        generator (torch.nn.Module): Trained generator model.

        rir_real (np.ndarray): 1D array of shape (L,) with the real RIR, where:

            - **L**: Length of the RIR signal.

        embedding (np.ndarray): 1D array of shape (F,) with the standardized embedding, where:

            - **F**: Feature dimension (usually 10).

        means (np.ndarray): 1D array of shape (F,) with the means used for standardization.

        stds (np.ndarray): 1D array of shape (F,) with the standard deviations used for standardization.

        z_dim (int): Dimensionality of the noise input vector.

        device (torch.device): Target device (e.g., "cpu" or "cuda").

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:

            - **rir_real_aligned**: 1D array of shape (T,) representing the aligned real RIR.
            - **rir_generated_aligned**: 1D array of shape (T,) representing the aligned generated RIR.
            - **original_embedding**: 1D array of shape (F,) representing the de-standardized embedding.
    """
    generator.eval()
    with torch.no_grad():
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        z_noise = torch.randn(1, z_dim, dtype=torch.float32).to(device)
        rir_generated = generator(embedding_tensor, z_noise).squeeze().cpu().numpy()
        # De-standardize the embedding for visualization
        original_embedding = destandardize_data(embedding_tensor.cpu().numpy(), means, stds)

    # Generate and align a synthetic RIR for the current test sample
    rir_real_aligned, rir_generated_aligned, _ = align_rirs(rir_real, rir_generated)
    return rir_real_aligned, rir_generated_aligned, original_embedding

if __name__ == "__main__":
    # Parse command-line arguments for training configuration
    parser = argparse.ArgumentParser(description="Train CWGAN model for generating RIRs.")
    parser.add_argument("--cache_rir", type=str, required=True, help="Path to cached RIR data.")
    parser.add_argument("--cache_emb", type=str, required=True, help="Path to cached embeddings.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--z_dim", type=int, default=50, help="Dimensionality of noise vector.")
    parser.add_argument("--gen_lr", type=float, default=5e-5, help="Learning rate for generator.")
    parser.add_argument("--crit_lr", type=float, default=5e-5, help="Learning rate for critic.")
    args = parser.parse_args()

    # Load the training and testing data from cached files
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_train, emb_test, rirs_train, rirs_test, _, _, _, _, _, _ = load_data(args.cache_rir, args.cache_emb)
    train_dataset = RIRDataset(rirs=rirs_train, embeddings=emb_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    gen = Generator(z_dim=args.z_dim).to(device)
    crit = Critic().to(device)

    # Start the training process
    train_model(
        generator=gen,
        critic=crit,
        loader=train_loader,
        num_epochs=args.num_epochs,
        z_dim=args.z_dim,
        gen_lr=args.gen_lr,
        crit_lr=args.crit_lr,
        device=device
    )
