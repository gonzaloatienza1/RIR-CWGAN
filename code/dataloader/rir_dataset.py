import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class RIRDataset(Dataset):
    """
    Class for the Room Impulse Response (RIR) Dataset and their corresponding embeddings. This dataset is intended 
    for tasks involving audio processing or room acoustics modeling, where each RIR sample is paired with a feature embedding.
    """
    def __init__(self, rirs: List[List[float]], embeddings: List[List[float]]):
        """
        Initialize the RIRDataset.

        Args:
            rirs (List[List[float]]): A list containing RIR waveforms (1D sequences).
            embeddings (List[List[float]]): A list containing the corresponding embeddings.
        """
        self.rirs = rirs
        self.embeddings = embeddings

    def __len__(self) -> int:
        """
        Get the number of rirs in the dataset.

        Returns:
            int: The total number of RIR-embedding pairs.
        """
        return len(self.rirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the RIR and its corresponding embedding at the specified index.

        Args:
            idx (int): Index of the RIR to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the following elements:

                - **rir_tensor (torch.Tensor)**: A 2D tensor of shape (1, N), where:
                 
                    - **N**: The length of the RIR.
                - **embedding_tensor (torch.Tensor)**: A 1D tensor of shape (D,), where:
                
                    - **D**: The embedding dimension.
        """
        # Convert the RIR waveform to a PyTorch tensor and add a channel dimension
        rir = self.rirs[idx]
        embedding = self.embeddings[idx]
        rir_tensor = torch.tensor(rir, dtype=torch.float32).unsqueeze(0)
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        return rir_tensor, embedding_tensor

