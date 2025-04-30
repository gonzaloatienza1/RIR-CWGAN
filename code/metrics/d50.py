import torch

def D50(h: torch.Tensor, ne: int) -> torch.Tensor:
    """
    Compute the Early-to-Total Sound Energy Ratio (D50) metric.

    The D50 metric quantifies the proportion of early arriving sound energy (within the first `ne` samples)
    relative to the total energy of the Room Impulse Response (RIR). It is commonly used to assess
    speech intelligibility and clarity in room acoustics.

    Args:
        h (torch.Tensor): RIR signal tensor. Expected shape: (N,), where:
            
            - **N**: Length of the RIR signal.
        
        ne (int): Index representing the end of the early energy window (inclusive).

    Returns:
        torch.Tensor: A 0D tensor (scalar) representing the D50 value in decibels (dB).
    """
    # Ensure tensor is 1D
    h = h.squeeze()

    # Compute energy in early portion of the RIR
    early_energy = torch.sum(h[:ne + 1] ** 2)

    # Compute total energy
    total_energy = torch.sum(h ** 2)

    # Compute D50 in decibels
    d50 = 10 * torch.log10(early_energy / total_energy)

    return d50
