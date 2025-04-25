import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import interp1d, CubicSpline


class BLNM(nn.Module):
    """
    Builds a Branched Latent Neural Map with (`num_inps_1` + `num_inps_2`) inputs and `num_outputs` outputs.
    The number of neurons per hidden layer are specified in the `neurons_per_layer` list.
    The number of hidden layers where the first and second inputs are separated is specified by `disentanglement_level`.
    """
    def __init__(self, neurons_per_layer, num_inps_1, num_inps_2, num_outs, disentanglement_level):
        super(BLNM, self).__init__()
        
        n_layers = len(neurons_per_layer)

        if disentanglement_level < 1 or disentanglement_level > n_layers:
            raise ValueError("The disentanglement level must be between 1 and the total number of hidden layers")

        self.disentanglement_level = disentanglement_level
        self.neurons_per_layer = neurons_per_layer

        # split between two branches
        BLNM_1_architecture = nn.ModuleList()
        BLNM_2_architecture = nn.ModuleList()

        inps_1 = num_inps_1
        inps_2 = num_inps_2


        for index in range(disentanglement_level):
            branch_inps_1 = neurons_per_layer[index] // 2
            branch_inps_2 = neurons_per_layer[index] - branch_inps_1

            # First branch
            BLNM_1_architecture.extend([
                nn.Linear(inps_1, branch_inps_1),
                nn.Tanh()
            ])

            # Second branch
            BLNM_2_architecture.extend([
                nn.Linear(inps_2, branch_inps_2),
                nn.Tanh()
            ])

            inps_1 = branch_inps_1
            inps_2 = branch_inps_2

        self.branch1 = nn.Sequential(*BLNM_1_architecture)
        self.branch2 = nn.Sequential(*BLNM_2_architecture)

        # define remaining layers after branches merge 
        combined_layers = nn.ModuleList()
        combined_input_size = neurons_per_layer[disentanglement_level - 1]
        for index in range(disentanglement_level, n_layers):
            combined_layers.extend([
                nn.Linear(combined_input_size, neurons_per_layer[index]),
                nn.Tanh(),
            ])
            combined_input_size = neurons_per_layer[index]
        
        # output layer 
        combined_layers.append(nn.Linear(combined_input_size, num_outs))
        self.combined = nn.Sequential(*combined_layers)

    def forward(self, x1, x2):
        chain_1 = self.branch1(x1)
        chain_2 = self.branch2(x2)

        # combine two branches 
        combine_chains = torch.cat((chain_1, chain_2), dim=2)
        output = self.combined(combine_chains) 
        
        return output
    
    def calculate_loss(model, space_adim, z_scores, AT_adim):
        model_output = model(z_scores, space_adim)[:, :, 0:1]
        loss = F.mse_loss(model_output, AT_adim)
        return loss


    def adimensionalize(data, data_min, data_max):
        return (data - data_min) / (data_max - data_min) 

    def dimensionalize(adim_data, data_min, data_max):
        data_min = torch.tensor(data_min, dtype = adim_data.dtype, device = adim_data.device)
        data_max = torch.tensor(data_max, dtype = adim_data.dtype, device = adim_data.device)
        return adim_data * (data_max - data_min) + data_min
    
