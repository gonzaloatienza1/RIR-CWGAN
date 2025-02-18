""" 
This script defines the core neural network models used in the CWGAN for generating Room Impulse Responses (RIRs).
It includes classes for the generator and critic models, along with auxiliary modules for conditioning and embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditioningAugmentation(nn.Module):
    """
    Input:
    - E: Input embedding of shape (batch_size, embedding_dim).
    
    Output:
    - C: Conditioned vector sampled using the reparameterization trick.
    - phi: Intermediate vector before sampling, containing concatenated mean and log-variance.
    """
    def __init__(self):
        super(ConditioningAugmentation, self).__init__()
        # Transform input embedding to a higher dimensional representation
        self.dense = nn.Linear(10, 256)  

    def forward(self, E):
        phi = F.leaky_relu(self.dense(E))  
        mean = phi[:, :128] 
        std = torch.exp(phi[:, 128:])  
        epsilon = torch.randn_like(std)  
        C = mean + epsilon * std 
        return C, phi  

class EmbeddingModule(nn.Module):
    """
    Input:
    - E: Input embedding of shape (batch_size, 10).
    
    Output:
    - Transformed embedding of shape (batch_size, 128).
    """
    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.dense = nn.Linear(10, 128) 

    def forward(self, E):
        return F.relu(self.dense(E))  

class Generator(nn.Module):
    """    
    Input:
    - E: Input embedding (batch_size, embedding_dim).
    - Z: Noise vector (batch_size, z_dim).
    
    Output:
    - Generated RIR of shape (batch_size, 1, 6144).
    """
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.ca = ConditioningAugmentation()  # Conditioning module
        self.fc1 = nn.Linear(128 + z_dim, 1024)  # Initial fully connected layer

        # Convolutional blocks for upsampling
        self.conv_blocks = nn.Sequential(
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

        self.flatten = nn.Flatten() 
        self.fc2 = nn.Sequential(nn.Linear(6907, 6144))  

    def forward(self, E, Z):
        C, _ = self.ca(E)  
        gen_input = torch.cat([C, Z], dim=1)  
        x = self.fc1(gen_input)  
        x = x.view(-1, 1024, 1)  
        x = self.conv_blocks(x)  
        flat = self.flatten(x)  
        fc = self.fc2(flat).unsqueeze(1)  
        return torch.tanh(fc)  

class Critic(nn.Module):  
    """
    Input:
    - RIR: Room Impulse Response (batch_size, 1, sequence_length).
    - E: Input embedding (batch_size, embedding_dim).
    
    Output:
    - Score representing the quality of the RIR.
    """
    def __init__(self):
        super(Critic, self).__init__()
        self.EmbMod = EmbeddingModule()  # Embedding module to process input embedding

        # Convolutional blocks for processing the RIR
        self.conv_blocks = nn.Sequential(
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

        self.flatten = nn.Flatten()  
        self.topmodel = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.Linear(1024, 512)
        )
        # Final layer to produce the scalar score
        self.fc = nn.Linear(640, 1)  

    def forward(self, RIR, E):
        x = self.conv_blocks(RIR)  
        flat = self.flatten(x)  
        x = self.topmodel(flat)  
        T = self.EmbMod(E)  
        x = torch.cat((x, T), dim=1)  
        fc = self.fc(x)  
        return fc.squeeze()  