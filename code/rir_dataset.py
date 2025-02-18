import torch
from torch.utils.data import Dataset

class RIRDataset(Dataset):
    """
    Custom dataset to handle RIRs and embeddings.
    rirs: List of room impulse responses.
    embeddings: Corresponding embeddings for each RIR.
    """
    def __init__(self, rirs, embeddings):
        self.rirs = rirs
        self.embeddings = embeddings

    def __len__(self):
        return len(self.rirs)

    def __getitem__(self, idx):
        rir = self.rirs[idx]
        embedding = self.embeddings[idx]
        rir_tensor = torch.tensor(rir, dtype=torch.float32).unsqueeze(0)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        return rir_tensor, embedding_tensor