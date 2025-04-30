import torch

def NMSEdB(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Normalized Mean Squared Error (NMSE) in decibels (dB).

    This metric compares the mean squared error between a predicted Room Impulse Response (RIR)
    and the ground truth RIR, normalized by the power of the ground truth. Expressing this ratio
    in decibels allows better interpretability in audio and acoustic applications.

    Args:
        y_pred (torch.Tensor): Predicted RIR tensor. Expected shape: (N,), where:
            
            - **N**: Length of the RIR signal.
            
        y_true (torch.Tensor): Ground truth RIR tensor. Expected shape: (N,), where:
            
            - **N**: Length of the RIR signal.

    Returns:
        torch.Tensor: A 0D tensor (scalar) representing the NMSE in decibels (dB).
    """
    # Compute mean squared error between predicted and true RIR
    mse = torch.mean((y_pred - y_true) ** 2)

    # Normalize by energy of the ground truth signal
    nmse_value = 10 * torch.log10(mse / torch.mean(y_true ** 2))

    return nmse_value