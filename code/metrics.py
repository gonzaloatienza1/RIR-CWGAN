import torch
import torch.nn as nn
import numpy as np

class NMSEdBLoss(nn.Module):
    """
    Custom loss class to calculate NMSE (in dB) between predicted and ground truth RIRs.
    y_pred: Predicted RIR.
    y_true: Ground truth RIR.
    """
    def __init__(self):
        super(NMSEdBLoss, self).__init__()

    def forward(self, y_pred, y_true):
        mse = torch.mean((y_pred - y_true)**2)
        nmse_value = 10 * torch.log10(mse / torch.mean(y_true**2))
        return nmse_value

class NPM(nn.Module):
    """
    Normalized Projection Misalignment (NPM) metric.
    Measures the misalignment between the predicted and true RIRs based on their projection.
    y_pred: Predicted RIR.
    y_true: Ground truth RIR.
    """
    def __init__(self):
        super(NPM, self).__init__()

    def forward(self, y_pred, y_true):
        norm_h_true_squared = torch.norm(y_true, p=2)**2
        beta = torch.dot(y_true.squeeze(), y_pred.squeeze()) / torch.norm(y_pred.squeeze(), p=2)**2
        squared_distance = torch.norm(y_true - beta * y_pred, p=2)**2
        npm = 10 * torch.log10(squared_distance / norm_h_true_squared)
        return npm

class DRR(nn.Module):
    """
    Direct-to-Reverberant Ratio (DRR) metric.
    Measures the ratio of direct path energy to reverberant energy in the RIR.
    h: Room impulse response (RIR).
    nd: Index representing the end of the direct path.
    """
    def __init__(self):
        super(DRR, self).__init__()

    def forward(self, h, nd):
        h = h.squeeze()
        # Calculate direct path energy up to index nd
        direct_path_energy = torch.sum(h[:nd + 1]**2)
        # Calculate reverberant energy after index nd
        reverberant_energy = torch.sum(h[nd + 1:]**2)
        drr = 10 * torch.log10(direct_path_energy / reverberant_energy)
        return drr

class D50(nn.Module):
    """
    Early-to-Total Sound Energy Ratio (D50) metric.
    Measures the ratio of early energy to total energy in the RIR.
    h: Room impulse response (RIR).
    ne: Index representing the end of the early energy window.
    """
    def __init__(self):
        super(D50, self).__init__()

    def forward(self, h, ne):
        h = h.squeeze()
        # Calculate early energy up to index ne
        early_energy = torch.sum(h[:ne + 1]**2)
        # Calculate total energy of the RIR
        total_energy = torch.sum(h[:]**2)
        D50 = 10 * torch.log10(early_energy / total_energy)
        return D50

class RMSE:
    """
    Root Mean Square Error (RMSE) metric.
    Measures the root mean square difference between two scalar values.
    yhat: Predicted value.
    y: True value.
    eps: Small epsilon to prevent division by zero.
    """
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, yhat, y):
        yhat, y = float(yhat), float(y)
        mse = np.mean((yhat - y) ** 2)
        return np.sqrt(mse + self.eps)