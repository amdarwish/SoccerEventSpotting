# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 09:48:40 2022

@author: The code is originally owned by https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Task1-ActionSpotting/TemporallyAwarePooling

It was modified to fit our new model
"""


import __future__

import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

class STE_t4(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(STE_t4, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size
        
        
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(int(self.input_size), 512)

            input_size = 512
            self.input_size = 512


        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.batch_conv1 = nn.BatchNorm2d(num_features=128, momentum=0.1,eps=1e-05) 


        # Encoder Layers 
        self.conv_e_1a  = nn.Conv1d(in_channels=128,    out_channels=256,   padding= 1, kernel_size=3)
        self.batch_e_1a = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 

        self.conv_e_2a  = nn.Conv1d(in_channels=256,   out_channels=512,   padding= 1, kernel_size=3)
        self.batch_e_2a = nn.BatchNorm1d(num_features=512, momentum=0.1,eps=1e-05) 


        self.maxpool   = nn.MaxPool1d (kernel_size = 2) 
               

        self.conv_3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=int(self.window_size_frame/2))
        self.batch_conv3 = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 
        
        self.fc_a = nn.Linear(256, 256)            
        self.fc_b = nn.Linear(256, self.num_classes+1)
        self.fc_f_a = nn.Linear(256, 256)
        self.fc_f_b = nn.Linear(256, self.window_size_frame)

        self.dropfc1    = nn.Dropout(p=0.5)
        self.drop_conv1 = nn.Dropout(p=0.2)
        self.drop_conv2 = nn.Dropout(p=0.2)
        
        self.drop1  = nn.Dropout(p=0.15)
        self.drop2  = nn.Dropout(p=0.2)

        

        self.drop_conv3 = nn.Dropout(p=0.3)        
        self.dropfc2_a = nn.Dropout(p=0.5)
        self.dropfc2_b = nn.Dropout(p=0.5)
        
        self.dropfc2_f_a = nn.Dropout(p=0.5)
        self.dropfc2_f_b = nn.Dropout(p=0.5)
        
        self.sigm = nn.Sigmoid()

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if(weights is not None):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(weights, checkpoint['epoch']))

    def forward(self, inputs):



        BS, FR, IC = inputs.shape
        if not IC == 512:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = F.relu(self.feature_extractor(self.dropfc1(inputs)))
            inputs = inputs.reshape(BS, FR, -1)
        
        # add a dimension
        inputs = inputs[:,None,:,:]

        # Base Convolutional Layers
        conv_1 = self.drop_conv1(F.relu(self.batch_conv1(self.conv_1(inputs))))

        tunet_inp   = torch.squeeze(conv_1,-1)


        
        conv_e_1a = self.drop1(F.relu(self.batch_e_1a(self.conv_e_1a(tunet_inp))))
        conv_e_1c = self.maxpool(conv_e_1a)
        
        conv_e_2a = self.drop2(F.relu(self.batch_e_2a(self.conv_e_2a(conv_e_1c))))

        
        
        conv_3 = self.drop_conv3(F.relu(self.batch_conv3(self.conv_3(conv_e_2a))))
        
        squeeze   = torch.squeeze(conv_3,-1)
        
        # Extra FC layer with dropout and sigmoid activation
        output1_a = F.relu   (self.fc_a(self.dropfc2_a(squeeze)))
        output1_b = self.sigm(self.fc_b(self.dropfc2_b(output1_a)))
        
        output2_a = F.relu   (self.fc_f_a(self.dropfc2_f_a(squeeze)))
        output2_b = self.sigm(self.fc_f_b(self.dropfc2_f_b(output2_a)))

        output = torch.cat((output1_b, output2_b),-1)

        
     #   print("output max. : ", torch.max(output))

        
        return output


if __name__ == "__main__":
    BS =256
    T = 15
    framerate= 2
    D = 512
    pool = "NetRVLAD++"
    model = Model(pool=pool, input_size=D, framerate=framerate, window_size=T)
    print(model)
    inp = torch.rand([BS,T*framerate,D])
    print(inp.shape)
    output = model(inp)
    print("The output shape: ", output.shape)


    