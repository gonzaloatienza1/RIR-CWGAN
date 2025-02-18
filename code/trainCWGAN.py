""" 
This script trains a Conditional Wasserstein Generative Adversarial Network (CWGAN) for generating Room Impulse Responses (RIRs).
The training process involves alternating between optimizing the generator and critic networks using the Wasserstein loss.
"""

import torch
import torch.optim as optim
import os
import time
import argparse
from model import Generator, Critic
from rir_dataset import RIRDataset
from torch.utils.data import DataLoader
from utils import load_data

# Set device to GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:
    """
    This class manages the training process of the CWGAN, including the generator and critic models.
    z_dim: Dimensionality of the noise vector.
    """
    def __init__(self, z_dim):
        # Initialize generator and critic models
        self.generator = Generator(z_dim).to(device)
        self.critic = Critic().to(device)

    def train(self, train_loader, num_epochs, z_dim, gen_lr, crit_lr):
        """
        Train the CWGAN model using the provided data.
        train_loader: DataLoader containing RIRs and embeddings for training.
        num_epochs: Number of training epochs.
        z_dim: Dimensionality of the noise vector.
        gen_lr: Learning rate for the generator.
        crit_lr: Learning rate for the critic.
        """
        # Optimizers and learning rate schedulers
        gen_optimizer = optim.RMSprop(self.generator.parameters(), lr=gen_lr)
        gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)
        critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=crit_lr)
        critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=50, gamma=0.5)

        clip_value = 0.01  # Weight clipping for critic
        critic_iterations = 5  # Train critic multiple times per generator step

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}:\n [", end="")
            start_time = time.time()
            gen_loss_log, critic_loss_log = [], []

            for rirs, embeddings in train_loader:
                # Move data to the specified device
                rirs, embeddings = rirs.to(device), embeddings.to(device)
                batch_size = rirs.size(0)

                # Train the Critic
                for _ in range(critic_iterations):
                    critic_optimizer.zero_grad()
                    # Calculate critic loss using real and generated RIRs
                    real_outputs = self.critic(rirs, embeddings)
                    fake_rirs = self.generator(embeddings, torch.randn(batch_size, z_dim, device=device))
                    fake_outputs = self.critic(fake_rirs.detach(), embeddings)
                    d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs)
                    d_loss.backward()
                    critic_optimizer.step()

                    # Clip weights to enforce Lipschitz constraint
                    for p in self.critic.parameters():
                        p.data.clamp_(-clip_value, clip_value)

                # Train the Generator
                gen_optimizer.zero_grad()
                fake_rirs = self.generator(embeddings, torch.randn(batch_size, z_dim, device=device))
                g_loss = -torch.mean(self.critic(fake_rirs, embeddings))  # Generator loss
                g_loss.backward()
                gen_optimizer.step()

                # Log losses for reporting
                gen_loss_log.append(g_loss.item())
                critic_loss_log.append(d_loss.item())

            # Step learning rate schedulers
            gen_scheduler.step()
            critic_scheduler.step()
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Training generator_loss:   {sum(gen_loss_log)/len(gen_loss_log):.4f} - Training critic_loss:   {sum(critic_loss_log)/len(critic_loss_log):.4f} - epoch_time: {epoch_time:.2f} s")

            # Save the generator model every 5 epochs
            if (epoch + 1) % 5 == 0:
                os.makedirs("CWGAN", exist_ok=True)
                torch.save(self.generator.state_dict(), f"CWGAN/generator_epch{epoch + 1}.pth")

if __name__ == "__main__":
    # Command-line arguments for training configuration
    parser = argparse.ArgumentParser(description="Train CWGAN model for generating RIRs.")
    parser.add_argument("--cache_rir", type=str, required=True, help="Path to cached RIR data.")
    parser.add_argument("--cache_emb", type=str, required=True, help="Path to cached embeddings.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--z_dim", type=int, default=50, help="Dimensionality of noise vector.")
    parser.add_argument("--gen_lr", type=float, default=5e-5, help="Learning rate for generator.")
    parser.add_argument("--crit_lr", type=float, default=5e-5, help="Learning rate for critic.")
    args = parser.parse_args()

    # Load data directly split into training and validation sets
    emb_train, emb_test, rirs_train, rirs_test, _, _, _, _ = load_data(args.cache_rir, args.cache_emb)

    # Prepare DataLoader
    train_dataset = RIRDataset(rirs=rirs_train, embeddings=emb_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize and train the model
    trainer = ModelTrainer(z_dim=args.z_dim)
    trainer.train(train_loader, num_epochs=args.num_epochs, z_dim=args.z_dim, gen_lr=args.gen_lr, crit_lr=args.crit_lr)
