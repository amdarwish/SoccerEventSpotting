# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 09:48:40 2022

@author: The code is originally owned by https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Task1-ActionSpotting/TemporallyAwarePooling

It was modified to fit our new model
"""

######################################################################################################

#The code is originally owned by https://github.com/SilvioGiancola/SoccerNetv2-DevKit/tree/main/Task1-ActionSpotting/TemporallyAwarePooling
#It was modified to fit our new model
######################################################################################################

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

        # Robust weight initialization to prevent gradient explosion
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

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
            # NaN and value check after feature extractor
            if torch.isnan(inputs).any():
                print("NaN detected after feature extractor")
            # print(f"inputs after feature extractor: min={inputs.min().item()}, max={inputs.max().item()}")
        # add a dimension
        inputs = inputs[:,None,:,:]
        # NaN and value check after adding dimension
        if torch.isnan(inputs).any():
            print("NaN detected after adding dimension")
        # print(f"inputs after add dim: min={inputs.min().item()}, max={inputs.max().item()}")
        # Base Convolutional Layers
        conv_1 = self.drop_conv1(F.relu(self.batch_conv1(self.conv_1(inputs))))
        # NaN and value check after conv_1
        if torch.isnan(conv_1).any():
            print("NaN detected after conv_1")
        # print(f"conv_1: min={conv_1.min().item()}, max={conv_1.max().item()}")
        tunet_inp   = torch.squeeze(conv_1,-1)
        # NaN and value check after tunet_inp
        if torch.isnan(tunet_inp).any():
            print("NaN detected after tunet_inp")
        # print(f"tunet_inp: min={tunet_inp.min().item()}, max={tunet_inp.max().item()}")
        conv_e_1a = self.drop1(F.relu(self.batch_e_1a(self.conv_e_1a(tunet_inp))))
        # NaN and value check after conv_e_1a
        if torch.isnan(conv_e_1a).any():
            print("NaN detected after conv_e_1a")
        # print(f"conv_e_1a: min={conv_e_1a.min().item()}, max={conv_e_1a.max().item()}")
        conv_e_1c = self.maxpool(conv_e_1a)
        # NaN and value check after conv_e_1c
        if torch.isnan(conv_e_1c).any():
            print("NaN detected after conv_e_1c")
        # print(f"conv_e_1c: min={conv_e_1c.min().item()}, max={conv_e_1c.max().item()}")
        conv_e_2a = self.drop2(F.relu(self.batch_e_2a(self.conv_e_2a(conv_e_1c))))
        # NaN and value check after conv_e_2a
        if torch.isnan(conv_e_2a).any():
            print("NaN detected after conv_e_2a")
        # print(f"conv_e_2a: min={conv_e_2a.min().item()}, max={conv_e_2a.max().item()}")
        conv_3 = self.drop_conv3(F.relu(self.batch_conv3(self.conv_3(conv_e_2a))))
        # NaN and value check after conv_3
        if torch.isnan(conv_3).any():
            print("NaN detected after conv_3")
        # print(f"conv_3: min={conv_3.min().item()}, max={conv_3.max().item()}")
        squeeze   = torch.squeeze(conv_3,-1)
        # NaN and value check after squeeze
        if torch.isnan(squeeze).any():
            print("NaN detected after squeeze")
        # print(f"squeeze: min={squeeze.min().item()}, max={squeeze.max().item()}")
        # Extra FC layer with dropout and sigmoid activation
        output1_a = F.relu   (self.fc_a(self.dropfc2_a(squeeze)))
        # NaN and value check after output1_a
        if torch.isnan(output1_a).any():
            print("NaN detected after output1_a")
        # print(f"output1_a: min={output1_a.min().item()}, max={output1_a.max().item()}")
        output1_b = self.sigm(self.fc_b(self.dropfc2_b(output1_a)))
        # NaN and value check after output1_b
        if torch.isnan(output1_b).any():
            print("NaN detected after output1_b")
        # print(f"output1_b: min={output1_b.min().item()}, max={output1_b.max().item()}")
        output2_a = F.relu   (self.fc_f_a(self.dropfc2_f_a(squeeze)))
        # NaN and value check after output2_a
        if torch.isnan(output2_a).any():
            print("NaN detected after output2_a")
        # print(f"output2_a: min={output2_a.min().item()}, max={output2_a.max().item()}")
        output2_b = self.sigm(self.fc_f_b(self.dropfc2_f_b(output2_a)))
        # NaN and value check after output2_b
        if torch.isnan(output2_b).any():
            print("NaN detected after output2_b")
        # print(f"output2_b: min={output2_b.min().item()}, max={output2_b.max().item()}")
        output = torch.cat((output1_b, output2_b),-1)
        # NaN and value check after final output
        if torch.isnan(output).any():
            print("NaN detected after final output")
        # print(f"final output: min={output.min().item()}, max={output.max().item()}")
        #   print("output max. : ", torch.max(output))
        return output



class STE_t2(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(STE_t2, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size
        
        
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(int(self.input_size), 512)
  #          self.feature_extractor = nn.Linear(int(self.input_size/4), 512)

            input_size = 512
            self.input_size = 512


        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.batch_conv1 = nn.BatchNorm2d(num_features=128, momentum=0.1,eps=1e-05) 
     #   self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))
     #   self.batch_conv2 = nn.BatchNorm2d(num_features=32, momentum=0.1,eps=1e-05) 

        # Encoder Layers 
        self.conv_e_1a  = nn.Conv1d(in_channels=128,    out_channels=256,   padding= 1, kernel_size=3)
        self.batch_e_1a = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 

        self.conv_e_2a  = nn.Conv1d(in_channels=256,   out_channels=512,   padding= 1, kernel_size=3)
        self.batch_e_2a = nn.BatchNorm1d(num_features=512, momentum=0.1,eps=1e-05) 


        self.maxpool   = nn.MaxPool1d (kernel_size = 2) 
               

        self.conv_3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=int(self.window_size_frame/2))
        self.batch_conv3 = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 
        
            
        self.fc = nn.Linear(256, self.num_classes+1)
        self.fc_f = nn.Linear(256, self.window_size_frame)

        self.dropfc1    = nn.Dropout(p=0.5)
        self.drop_conv1 = nn.Dropout(p=0.2)
        self.drop_conv2 = nn.Dropout(p=0.2)
        
        self.drop1  = nn.Dropout(p=0.15)
        self.drop2  = nn.Dropout(p=0.2)
      #  self.drop3  = nn.Dropout(p=0.25)
      #  self.drop4  = nn.Dropout(p=0.3)
        

        self.drop_conv3 = nn.Dropout(p=0.3)        
        self.dropfc2 = nn.Dropout(p=0.5)
        self.dropfc2_f = nn.Dropout(p=0.5)
        
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
        # input_shape: (batch,frames,dim_features)


       # inputs = self.maxpool(inputs)
       # inputs = self.maxpool(inputs)


        BS, FR, IC = inputs.shape
        if not IC == 512:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = F.relu(self.feature_extractor(self.dropfc1(inputs)))
            inputs = inputs.reshape(BS, FR, -1)
        
        # add a dimension
        inputs = inputs[:,None,:,:]

        # Base Convolutional Layers
        conv_1 = self.drop_conv1(F.relu(self.batch_conv1(self.conv_1(inputs))))
      #  conv_2 = self.drop_conv2(F.relu(self.batch_conv2(self.conv_2(conv_1))))

        tunet_inp   = torch.squeeze(conv_1,-1)


        
        conv_e_1a = self.drop1(F.relu(self.batch_e_1a(self.conv_e_1a(tunet_inp))))
        conv_e_1c = self.maxpool(conv_e_1a)
        
        conv_e_2a = self.drop2(F.relu(self.batch_e_2a(self.conv_e_2a(conv_e_1c))))
     #   conv_e_2c = self.maxpool(conv_e_2a)

     #   conv_e_3a = self.drop3(F.relu(self.batch_e_3a(self.conv_e_3a(conv_e_2c))))
     #   conv_e_3c = self.maxpool(conv_e_3a)
        
     #   conv_e_4a = self.drop4(F.relu(self.batch_e_4a(self.conv_e_4a(conv_e_3c))))
        
        
        conv_3 = self.drop_conv3(F.relu(self.batch_conv3(self.conv_3(conv_e_2a))))
        
        squeeze   = torch.squeeze(conv_3,-1)
        
        # Extra FC layer with dropout and sigmoid activation
        output1 = self.sigm(self.fc(self.dropfc2(squeeze)))
        output2 = self.sigm(self.fc_f(self.dropfc2_f(squeeze)))

        output = torch.cat((output1, output2),-1)

        
     #   print("output max. : ", torch.max(output))

        
        return output


class STE_t1(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(STE_t1, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size
        
        
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(int(self.input_size/4), 512)
            input_size = 512
            self.input_size = 512


        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.batch_conv1 = nn.BatchNorm2d(num_features=128, momentum=0.1,eps=1e-05) 
     #   self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))
     #   self.batch_conv2 = nn.BatchNorm2d(num_features=32, momentum=0.1,eps=1e-05) 

        # Encoder Layers 
        self.conv_e_1a  = nn.Conv1d(in_channels=128,    out_channels=256,   padding= 1, kernel_size=3)
        self.batch_e_1a = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 

        self.conv_e_2a  = nn.Conv1d(in_channels=256,   out_channels=512,   padding= 1, kernel_size=3)
        self.batch_e_2a = nn.BatchNorm1d(num_features=512, momentum=0.1,eps=1e-05) 


        self.maxpool   = nn.MaxPool1d (kernel_size = 2) 
               

        self.conv_3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=int(self.window_size_frame/2))
        self.batch_conv3 = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 
        
            
        self.fc = nn.Linear(256, self.num_classes+1)

        self.dropfc1    = nn.Dropout(p=0.5)
        self.drop_conv1 = nn.Dropout(p=0.2)
        self.drop_conv2 = nn.Dropout(p=0.2)
        
        self.drop1  = nn.Dropout(p=0.15)
        self.drop2  = nn.Dropout(p=0.2)
      #  self.drop3  = nn.Dropout(p=0.25)
      #  self.drop4  = nn.Dropout(p=0.3)
        

        self.drop_conv3 = nn.Dropout(p=0.3)        
        self.dropfc2 = nn.Dropout(p=0.5)
        
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
        # input_shape: (batch,frames,dim_features)


        inputs = self.maxpool(inputs)
        inputs = self.maxpool(inputs)


        BS, FR, IC = inputs.shape
        if not IC == 512:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = F.relu(self.feature_extractor(self.dropfc1(inputs)))
            inputs = inputs.reshape(BS, FR, -1)
        
        # add a dimension
        inputs = inputs[:,None,:,:]

        # Base Convolutional Layers
        conv_1 = self.drop_conv1(F.relu(self.batch_conv1(self.conv_1(inputs))))
      #  conv_2 = self.drop_conv2(F.relu(self.batch_conv2(self.conv_2(conv_1))))

        tunet_inp   = torch.squeeze(conv_1,-1)


        
        conv_e_1a = self.drop1(F.relu(self.batch_e_1a(self.conv_e_1a(tunet_inp))))
        conv_e_1c = self.maxpool(conv_e_1a)
        
        conv_e_2a = self.drop2(F.relu(self.batch_e_2a(self.conv_e_2a(conv_e_1c))))
     #   conv_e_2c = self.maxpool(conv_e_2a)

     #   conv_e_3a = self.drop3(F.relu(self.batch_e_3a(self.conv_e_3a(conv_e_2c))))
     #   conv_e_3c = self.maxpool(conv_e_3a)
        
     #   conv_e_4a = self.drop4(F.relu(self.batch_e_4a(self.conv_e_4a(conv_e_3c))))
        
        
        conv_3 = self.drop_conv3(F.relu(self.batch_conv3(self.conv_3(conv_e_2a))))
        
        squeeze   = torch.squeeze(conv_3,-1)
        
        # Extra FC layer with dropout and sigmoid activation
        output = self.sigm(self.fc(self.dropfc2(squeeze)))
     #   print("output max. : ", torch.max(output))

        
        return output


class MSencoding_Baidu3(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(MSencoding_Baidu3, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size
        
        
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(int(self.input_size/4), 512)
            input_size = 512
            self.input_size = 512


        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.batch_conv1 = nn.BatchNorm2d(num_features=128, momentum=0.1,eps=1e-05) 
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))
        self.batch_conv2 = nn.BatchNorm2d(num_features=32, momentum=0.1,eps=1e-05) 

        # Encoder Layers 
        self.conv_e_1a  = nn.Conv1d(in_channels=32,    out_channels=64,   padding= 1, kernel_size=3)
        self.batch_e_1a = nn.BatchNorm1d(num_features=64, momentum=0.1,eps=1e-05) 

        self.conv_e_2a  = nn.Conv1d(in_channels=64,   out_channels=128,   padding= 1, kernel_size=3)
        self.batch_e_2a = nn.BatchNorm1d(num_features=128, momentum=0.1,eps=1e-05) 


        self.maxpool   = nn.MaxPool1d (kernel_size = 2) 
               

        self.conv_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=int(self.window_size_frame/2))
        self.batch_conv3 = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 
        
            
        self.fc = nn.Linear(256, self.num_classes+1)

        self.dropfc1    = nn.Dropout(p=0.5)
        self.drop_conv1 = nn.Dropout(p=0.2)
        self.drop_conv2 = nn.Dropout(p=0.2)
        
        self.drop1  = nn.Dropout(p=0.15)
        self.drop2  = nn.Dropout(p=0.2)
      #  self.drop3  = nn.Dropout(p=0.25)
      #  self.drop4  = nn.Dropout(p=0.3)
        

        self.drop_conv3 = nn.Dropout(p=0.3)        
        self.dropfc2 = nn.Dropout(p=0.5)
        
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
        # input_shape: (batch,frames,dim_features)


        inputs = self.maxpool(inputs)
        inputs = self.maxpool(inputs)


        BS, FR, IC = inputs.shape
        if not IC == 512:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = F.relu(self.feature_extractor(self.dropfc1(inputs)))
            inputs = inputs.reshape(BS, FR, -1)
        
        # add a dimension
        inputs = inputs[:,None,:,:]

        # Base Convolutional Layers
        conv_1 = self.drop_conv1(F.relu(self.batch_conv1(self.conv_1(inputs))))
        conv_2 = self.drop_conv2(F.relu(self.batch_conv2(self.conv_2(conv_1))))

        tunet_inp   = torch.squeeze(conv_2,-1)


        
        conv_e_1a = self.drop1(F.relu(self.batch_e_1a(self.conv_e_1a(tunet_inp))))
        conv_e_1c = self.maxpool(conv_e_1a)
        
        conv_e_2a = self.drop2(F.relu(self.batch_e_2a(self.conv_e_2a(conv_e_1c))))
     #   conv_e_2c = self.maxpool(conv_e_2a)

     #   conv_e_3a = self.drop3(F.relu(self.batch_e_3a(self.conv_e_3a(conv_e_2c))))
     #   conv_e_3c = self.maxpool(conv_e_3a)
        
     #   conv_e_4a = self.drop4(F.relu(self.batch_e_4a(self.conv_e_4a(conv_e_3c))))
        
        
        conv_3 = self.drop_conv3(F.relu(self.batch_conv3(self.conv_3(conv_e_2a))))
        
        squeeze   = torch.squeeze(conv_3,-1)
        
        # Extra FC layer with dropout and sigmoid activation
        output = self.sigm(self.fc(self.dropfc2(squeeze)))
     #   print("output max. : ", torch.max(output))

        
        return output


class MSencoding_Baidu2(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(MSencoding_Baidu2, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size
        
        
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(int(self.input_size/4), 512)
            input_size = 512
            self.input_size = 512


        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.batch_conv1 = nn.BatchNorm2d(num_features=128, momentum=0.1,eps=1e-05) 
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))
        self.batch_conv2 = nn.BatchNorm2d(num_features=32, momentum=0.1,eps=1e-05) 

        # Encoder Layers 
        self.conv_e_1a  = nn.Conv1d(in_channels=32,    out_channels=64,   padding= 1, kernel_size=3)
        self.batch_e_1a = nn.BatchNorm1d(num_features=64, momentum=0.1,eps=1e-05) 

        self.conv_e_2a  = nn.Conv1d(in_channels=64,   out_channels=128,   padding= 1, kernel_size=3)
        self.batch_e_2a = nn.BatchNorm1d(num_features=128, momentum=0.1,eps=1e-05) 

        self.conv_e_3a  = nn.Conv1d(in_channels=128,  out_channels=256,   padding= 1, kernel_size=3)
        self.batch_e_3a = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 


        self.maxpool   = nn.MaxPool1d (kernel_size = 2) 
               

        self.conv_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=int(self.window_size_frame/4))
        self.batch_conv3 = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 
        
            
        self.fc = nn.Linear(256, self.num_classes+1)

        self.dropfc1    = nn.Dropout(p=0.5)
        self.drop_conv1 = nn.Dropout(p=0.2)
        self.drop_conv2 = nn.Dropout(p=0.2)
        
        self.drop1  = nn.Dropout(p=0.15)
        self.drop2  = nn.Dropout(p=0.2)
        self.drop3  = nn.Dropout(p=0.25)
      #  self.drop4  = nn.Dropout(p=0.3)
        

        self.drop_conv3 = nn.Dropout(p=0.3)        
        self.dropfc2 = nn.Dropout(p=0.5)
        
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
        # input_shape: (batch,frames,dim_features)


        inputs = self.maxpool(inputs)
        inputs = self.maxpool(inputs)


        BS, FR, IC = inputs.shape
        if not IC == 512:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = F.relu(self.feature_extractor(self.dropfc1(inputs)))
            inputs = inputs.reshape(BS, FR, -1)
        
        # add a dimension
        inputs = inputs[:,None,:,:]

        # Base Convolutional Layers
        conv_1 = self.drop_conv1(F.relu(self.batch_conv1(self.conv_1(inputs))))
        conv_2 = self.drop_conv2(F.relu(self.batch_conv2(self.conv_2(conv_1))))

        tunet_inp   = torch.squeeze(conv_2,-1)


        
        conv_e_1a = self.drop1(F.relu(self.batch_e_1a(self.conv_e_1a(tunet_inp))))
        conv_e_1c = self.maxpool(conv_e_1a)
        
        conv_e_2a = self.drop2(F.relu(self.batch_e_2a(self.conv_e_2a(conv_e_1c))))
        conv_e_2c = self.maxpool(conv_e_2a)

        conv_e_3a = self.drop3(F.relu(self.batch_e_3a(self.conv_e_3a(conv_e_2c))))
     #   conv_e_3c = self.maxpool(conv_e_3a)
        
     #   conv_e_4a = self.drop4(F.relu(self.batch_e_4a(self.conv_e_4a(conv_e_3c))))
        
        
        conv_3 = self.drop_conv3(F.relu(self.batch_conv3(self.conv_3(conv_e_3a))))
        
        squeeze   = torch.squeeze(conv_3,-1)
        
        # Extra FC layer with dropout and sigmoid activation
        output = self.sigm(self.fc(self.dropfc2(squeeze)))
     #   print("output max. : ", torch.max(output))

        
        return output


class MSencoding_Baidu(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(MSencoding_Baidu, self).__init__()

        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size
        
        
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(int(self.input_size/4), 512)
            input_size = 512
            self.input_size = 512


        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.batch_conv1 = nn.BatchNorm2d(num_features=128, momentum=0.1,eps=1e-05) 
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))
        self.batch_conv2 = nn.BatchNorm2d(num_features=32, momentum=0.1,eps=1e-05) 

        # Encoder Layers 
        self.conv_e_1a  = nn.Conv1d(in_channels=32,    out_channels=64,   padding= 1, kernel_size=3)
        self.batch_e_1a = nn.BatchNorm1d(num_features=64, momentum=0.1,eps=1e-05) 

        self.conv_e_2a  = nn.Conv1d(in_channels=64,   out_channels=128,   padding= 1, kernel_size=3)
        self.batch_e_2a = nn.BatchNorm1d(num_features=128, momentum=0.1,eps=1e-05) 

        self.conv_e_3a  = nn.Conv1d(in_channels=128,  out_channels=256,   padding= 1, kernel_size=3)
        self.batch_e_3a = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 

        self.conv_e_4a  = nn.Conv1d(in_channels=256, out_channels=512,   padding= 1, kernel_size=3)
        self.batch_e_4a = nn.BatchNorm1d(num_features=512, momentum=0.1,eps=1e-05) 


        self.maxpool   = nn.MaxPool1d (kernel_size = 2) 
               

        self.conv_3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=int(self.window_size_frame/8))
        self.batch_conv3 = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 
        
            
        self.fc = nn.Linear(256, self.num_classes+1)

        self.dropfc1    = nn.Dropout(p=0.5)
        self.drop_conv1 = nn.Dropout(p=0.2)
        self.drop_conv2 = nn.Dropout(p=0.2)
        
        self.drop1  = nn.Dropout(p=0.15)
        self.drop2  = nn.Dropout(p=0.2)
        self.drop3  = nn.Dropout(p=0.25)
        self.drop4  = nn.Dropout(p=0.3)
        

        self.drop_conv3 = nn.Dropout(p=0.3)        
        self.dropfc2 = nn.Dropout(p=0.5)
        
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
        # input_shape: (batch,frames,dim_features)


        inputs = self.maxpool(inputs)
        inputs = self.maxpool(inputs)


        BS, FR, IC = inputs.shape
        if not IC == 512:
            inputs = inputs.reshape(BS*FR, IC)
            inputs = F.relu(self.feature_extractor(self.dropfc1(inputs)))
            inputs = inputs.reshape(BS, FR, -1)
        
        # add a dimension
        inputs = inputs[:,None,:,:]

        # Base Convolutional Layers
        conv_1 = self.drop_conv1(F.relu(self.batch_conv1(self.conv_1(inputs))))
        conv_2 = self.drop_conv2(F.relu(self.batch_conv2(self.conv_2(conv_1))))

        tunet_inp   = torch.squeeze(conv_2,-1)


        
        conv_e_1a = self.drop1(F.relu(self.batch_e_1a(self.conv_e_1a(tunet_inp))))
        conv_e_1c = self.maxpool(conv_e_1a)
        
        conv_e_2a = self.drop2(F.relu(self.batch_e_2a(self.conv_e_2a(conv_e_1c))))
        conv_e_2c = self.maxpool(conv_e_2a)

        conv_e_3a = self.drop3(F.relu(self.batch_e_3a(self.conv_e_3a(conv_e_2c))))
        conv_e_3c = self.maxpool(conv_e_3a)
        
        conv_e_4a = self.drop4(F.relu(self.batch_e_4a(self.conv_e_4a(conv_e_3c))))
        
        
        conv_3 = self.drop_conv3(F.relu(self.batch_conv3(self.conv_3(conv_e_4a))))
        
        squeeze   = torch.squeeze(conv_3,-1)
        
        # Extra FC layer with dropout and sigmoid activation
        output = self.sigm(self.fc(self.dropfc2(squeeze)))
     #   print("output max. : ", torch.max(output))

        
        return output


class MSencoding(nn.Module):
    def __init__(self, weights=None, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2, pool="NetVLAD"):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """
        super(MSencoding, self).__init__()
        self.window_size_frame=window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.pool = pool
        self.vlad_k = vocab_size
        # are feature alread PCA'ed?
        if not self.input_size == 512:   
            self.feature_extractor = nn.Linear(self.input_size, 512)
            input_size = 512
            self.input_size = 512
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1,input_size))
        self.batch_conv1 = nn.BatchNorm2d(num_features=128, momentum=0.1,eps=1e-05) 
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1))
        self.batch_conv2 = nn.BatchNorm2d(num_features=32, momentum=0.1,eps=1e-05) 
        # Encoder Layers 
        self.conv_e_1a  = nn.Conv1d(in_channels=32,    out_channels=64,   padding= 1, kernel_size=3)
        self.batch_e_1a = nn.BatchNorm1d(num_features=64, momentum=0.1,eps=1e-05) 
        self.conv_e_2a  = nn.Conv1d(in_channels=64,   out_channels=128,   padding= 1, kernel_size=3)
        self.batch_e_2a = nn.BatchNorm1d(num_features=128, momentum=0.1,eps=1e-05) 
        self.conv_e_3a  = nn.Conv1d(in_channels=128,  out_channels=256,   padding= 1, kernel_size=3)
        self.batch_e_3a = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 
        self.conv_e_4a  = nn.Conv1d(in_channels=256, out_channels=512,   padding= 1, kernel_size=3)
        self.batch_e_4a = nn.BatchNorm1d(num_features=512, momentum=0.1,eps=1e-05) 
        self.maxpool   = nn.MaxPool1d (kernel_size = 2) 
        self.conv_3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=int(self.window_size_frame/8))
        self.batch_conv3 = nn.BatchNorm1d(num_features=256, momentum=0.1,eps=1e-05) 
        self.fc = nn.Linear(256, self.num_classes+1)
        self.dropfc1    = nn.Dropout(p=0.5)
        self.drop_conv1 = nn.Dropout(p=0.2)
        self.drop_conv2 = nn.Dropout(p=0.2)
        self.drop1  = nn.Dropout(p=0.15)
        self.drop2  = nn.Dropout(p=0.2)
        self.drop3  = nn.Dropout(p=0.25)
        self.drop4  = nn.Dropout(p=0.3)
        self.drop_conv3 = nn.Dropout(p=0.3)        
        self.dropfc2 = nn.Dropout(p=0.5)
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
        inputs = inputs[:,None,:,:]
        conv_1 = self.drop_conv1(F.relu(self.batch_conv1(self.conv_1(inputs))))
        conv_2 = self.drop_conv2(F.relu(self.batch_conv2(self.conv_2(conv_1))))
        tunet_inp   = torch.squeeze(conv_2,-1)
        conv_e_1a = self.drop1(F.relu(self.batch_e_1a(self.conv_e_1a(tunet_inp))))
        conv_e_1c = self.maxpool(conv_e_1a)
        conv_e_2a = self.drop2(F.relu(self.batch_e_2a(self.conv_e_2a(conv_e_1c))))
        conv_e_2c = self.maxpool(conv_e_2a)
        conv_e_3a = self.drop3(F.relu(self.batch_e_3a(self.conv_e_3a(conv_e_2c))))
        conv_e_3c = self.maxpool(conv_e_3a)
        conv_e_4a = self.drop4(F.relu(self.batch_e_4a(self.conv_e_4a(conv_e_3c))))
        conv_3 = self.drop_conv3(F.relu(self.batch_conv3(self.conv_3(conv_e_4a))))
        squeeze   = torch.squeeze(conv_3,-1)
        output = self.sigm(self.fc(self.dropfc2(squeeze)))
        return output


if __name__ == "__main__":
    BS =256
    T = 15
    framerate= 2
    D = 512
    pool = "NetRVLAD++"
    model = MSencoding(pool=pool, input_size=D, framerate=framerate, window_size=T)
    print(model)
    inp = torch.rand([BS,T*framerate,D])
    print(inp.shape)
    output = model(inp)
    print("The output shape: ", output.shape)




