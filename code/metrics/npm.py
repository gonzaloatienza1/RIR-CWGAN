import torch

def NPM(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Compute the Normalized Projection Misalignment (NPM) metric.

    This metric evaluates how well the predicted Room Impulse Response (RIR) aligns with the ground truth,
    based on the projection of one onto the other. A lower value indicates better alignment. The output
    is expressed in decibels (dB).

    Args:
        y_pred (torch.Tensor): Predicted RIR tensor. Expected shape: (N,), where:
            
            - **N**: Length of the RIR signal.
            
        y_true (torch.Tensor): Ground truth RIR tensor.  Expected shape: (N,), where:
            
            - **N**: Length of the RIR signal.

    Returns:
        torch.Tensor: A 0D tensor (scalar) representing the NPM value in decibels (dB).
    """
    # Ensure both tensors are 1D
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()

    # Compute projection factor beta
    beta = torch.dot(y_true, y_pred) / torch.norm(y_pred, p=2)**2

    # Compute squared distance to projection
    squared_distance = torch.norm(y_true - beta * y_pred, p=2)**2

    # Normalize by the energy of the true RIR
    norm_h_true_squared = torch.norm(y_true, p=2)**2

    # Compute NPM in decibels
    npm = 10 * torch.log10(squared_distance / norm_h_true_squared)

    return npm
