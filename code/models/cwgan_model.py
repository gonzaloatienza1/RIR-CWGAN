"""
This script defines the neural network models used in a Conditional Wasserstein GAN (CWGAN)
for generating Room Impulse Responses (RIRs). It includes:
- ConditioningAugmentation: Applies reparameterization trick to embeddings.
- EmbeddingModule: Transforms embeddings into a higher-dimensional latent space.
- Generator: Generates RIRs from noise and conditioning.
- Critic: Evaluates the realism of generated RIRs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class _ConditioningAugmentation(nn.Module):
    """
    Module to apply conditioning augmentation to an input embedding by producing a
    latent vector.
    """
    def __init__(self) -> None:
        """
        Initializes the ConditioningAugmentation module.
        """
        super().__init__()
        self.dense: nn.Linear = nn.Linear(10, 256)

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the module.

        Args:
            E (torch.Tensor): A 2D tensor of shape (batch_size, 10) that represents the input embedding tensor, where:

                - **batch_size**: The number of samples per batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the following elements:

                - **C (torch.Tensor)**: A 2D tensor of shape (batch_size, 128) that represents the sampled latent conditioning vector.
                - **phi (torch.Tensor)**: A 2D tensor of shape (batch_size, 256) that represents the intermediate tensor that contains mean and log-variance.
        """
        phi = F.leaky_relu(self.dense(E))
        # Split the output into mean and standard deviation for reparameterization
        mean = phi[:, :128]
        std = torch.exp(phi[:, 128:])
        epsilon = torch.randn_like(std)
        # Sample the latent vector
        C = mean + epsilon * std
        return C, phi

class _EmbeddingModule(nn.Module):
    """
    Module to transform a 10-dimensional input embedding into a 128-dimensional latent vector.
    """
    def __init__(self) -> None:
        """
        Initializes the EmbeddingModule.
        """
        super().__init__()
        self.dense: nn.Linear = nn.Linear(10, 128)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the module.

        Args:
            E (torch.Tensor): A 2D tensor of shape (batch_size, 10) that represents the input tensor, where:
            
                - **batch_size**: The number of samples per batch.

        Returns:
            torch.Tensor: A 2D tensor of shape (batch_size, 128) that represents the output tensor.
        """
        return F.relu(self.dense(E))

class Generator(nn.Module):
    """
    Generator network that produces Room Impulse Responses (RIRs) conditioned on embeddings
    and driven by a noise vector.
    """
    def __init__(self, z_dim: int) -> None:
        """
        Initializes the Generator.

        Args:
            z_dim (int): Dimensionality of the noise vector.
        """
        super().__init__()
        self.ca: _ConditioningAugmentation = _ConditioningAugmentation()
        # Fully connected layer to combine noise and conditioning vector
        self.fc1: nn.Linear = nn.Linear(128 + z_dim, 1024)

        self.conv_blocks: nn.Sequential = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(512, 512, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(512, 512, kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(256, 256, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(256, 256, kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=4, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(64, 64, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(64, 64, kernel_size=9, stride=4, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(32, 32, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(32, 32, kernel_size=9, stride=4, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=9, stride=1, padding=1),
            nn.ConvTranspose1d(1, 1, kernel_size=9, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.flatten: nn.Flatten = nn.Flatten()
        # Fully connected layer to reshape the output to the desired RIR size
        self.fc2: nn.Sequential = nn.Sequential(nn.Linear(6907, 6144))

    def forward(self, E: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Generator.

        Args:
            E (torch.Tensor): A 2D tensor of shape (batch_size, 10) that represents the input embedding tensor, where:

                - **batch_size**: The number of samples per batch.
            
            Z (torch.Tensor): A 2D tensor of shape  (batch_size, z_dim) that represents the noise tensor, where:
                
                - **z_dim**: Dimensionality of the noise vector.

        Returns:
            torch.Tensor: A 3D tensor of shape (batch_size, 1, 6144) that represents the generated RIR.
        """
        C, _ = self.ca(E)
        gen_input = torch.cat([C, Z], dim=1)
        x = self.fc1(gen_input)
        # Reshape the output to match the input dimensions of the convolutional blocks
        x = x.view(-1, 1024, 1)
        x = self.conv_blocks(x)
        flat = self.flatten(x)
        fc = self.fc2(flat).unsqueeze(1)
        return torch.tanh(fc)

class Critic(nn.Module):
    """
    Critic network that evaluates the realism of RIRs based on the input RIR and embedding.
    """
    def __init__(self) -> None:
        """
        Initializes the Critic.
        """
        super().__init__()
        self.EmbMod: _EmbeddingModule = _EmbeddingModule()

        self.conv_blocks: nn.Sequential = nn.Sequential(
            nn.Conv1d(1, 64, 4, 1, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, 8, 8, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 8, 8, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, 16, 16, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )

        self.flatten: nn.Flatten = nn.Flatten()
        self.topmodel: nn.Sequential = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.Linear(1024, 512)
        )
        self.fc: nn.Linear = nn.Linear(640, 1)  # Final fully connected layer to output the critic score

    def forward(self, RIR: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Critic.

        Args:
            RIR (torch.Tensor):  A 3D tensor of shape (batch_size, 1, signal_length) that represents the input RIR tensor, where:

                - **batch_size**: The number of samples per batch.
                - **signal_length**: Number of samples per audio signal
            
            E (torch.Tensor): A 2D tensor of shape (batch_size, 10) that represents the input embedding tensor.

        Returns:
            torch.Tensor: A 1D tensor of shape (batch_size,) that represents the critic score as a scalar per batch sample.
        """
        x = self.conv_blocks(RIR)
        flat = self.flatten(x)
        x = self.topmodel(flat)
        T = self.EmbMod(E)
        x = torch.cat((x, T), dim=1)
        fc = self.fc(x)
        return fc.squeeze()