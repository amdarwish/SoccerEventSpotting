# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 09:47:54 2022

@author: The code is originally owned by https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Task1-ActionSpotting/TemporallyAwarePooling

It was modified to fit our new model
"""


import torch


class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, labels, output):
        
        try:
            # Convert to float (ensure precision consistency)
            labels = labels.float()
            output = output.float()
            
            # Check for invalid values before computation
            if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                raise ValueError("NaN or Inf detected in output tensor")

            if torch.any(torch.isnan(labels)) or torch.any(torch.isinf(labels)):
                raise ValueError("NaN or Inf detected in labels tensor")

            # Compute the xlogy terms safely
            term1 = -torch.special.xlogy(labels, output)
            term2 = -torch.special.xlogy(1.0 - labels, 1.0 - output)

            # Check for NaNs or Infs after computation
            if torch.any(torch.isnan(term1)) or torch.any(torch.isinf(term1)):
                raise ValueError(f"NaN or Inf detected in term1: {term1}")

            if torch.any(torch.isnan(term2)) or torch.any(torch.isinf(term2)):
                raise ValueError(f"NaN or Inf detected in term2: {term2}")

            # Compute the final loss
            loss = torch.mean(torch.mean(term1 + term2))

            # Check for final NaN or Inf
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN or Inf detected in loss: {loss}")

            return loss

        except Exception as e:
            print(f"Error in loss computation: {e}")
            # print(f"Labels: {labels}")
            # print(f"Output: {output}")
            return torch.tensor(float("nan"))  # Return NaN tensor to indicate failure

        