import torch

def DRR(h: torch.Tensor, nd: int) -> torch.Tensor:
    """
    Compute the Direct-to-Reverberant Ratio (DRR) metric.

    The DRR metric quantifies the ratio of direct path energy (from the start up to index `nd`) to
    the energy of the reverberant tail (from `nd + 1` to end) of a Room Impulse Response (RIR).
    This is used to evaluate acoustic clarity and speech intelligibility in room environments.

    Args:
        h (torch.Tensor): RIR signal tensor. Expected shape: (N,), where:
            
            - **N**: Length of the RIR signal.
        
        nd (int): Index representing the end of the direct path window (inclusive).

    Returns:
        torch.Tensor: A 0D tensor (scalar) representing the DRR value in decibels (dB).
    """
    # Ensure tensor is 1D
    h = h.squeeze()

    # Compute energy in direct path (early part of the RIR)
    direct_path_energy = torch.sum(h[:nd + 1] ** 2)

    # Compute energy in reverberant part (everything after nd)
    reverberant_energy = torch.sum(h[nd + 1:] ** 2)

    # Compute DRR in decibels
    drr = 10 * torch.log10(direct_path_energy / reverberant_energy)

    return drr