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
        
        return torch.mean(torch.mean(-torch.special.xlogy(labels.float(),output.float()) - torch.special.xlogy(1.0-labels.float(),1.0-output.float())))
        
        