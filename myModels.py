import os
import time
import pickle
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class lilyNN(nn.Module):
    '''
    lilyNN is a 3-layers NN to predict 2 orders' similarity 
    input: 2 orders' features (batch,features,2)
    output: their similarity (batch, 1)

    '''
    def __init__(self, feature_num, hidden_size, order_num=2):
        # super this class
        super(lilyNN, self).__init__()
        # define the size for the tensors
        self.input_size = order_num
        self.feature_num = feature_num
        self.hidden_size = hidden_size

        # 1st layer: nonlinear trans inside each feature, 
        # (batch,features,2) -> (batch,features,hidden_size) -> (batch,features,1)
        self.projecting_layer1 = nn.Sequential(nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(True), nn.Linear(self.hidden_size, 1))
        # 3rd layer:nonlinear trans over features, 
        # (batch,features) -> (batch,hidden_size) -> (batch,1)
        self.projecting_layer2 = nn.Sequential(nn.Linear(self.feature_num, self.hidden_size),
            nn.LeakyReLU(True), nn.Linear(self.hidden_size, 1))
    def forward(self, input):
        output = self.projecting_layer1(input) # (batch,features,2) -> (batch,features,1)
        output = output.squeeze(-1) # (batch,features,1) -> (batch,features)
        output = self.projecting_layer2(output) # (batch,features) -> (batch,1)
        output = torch.sigmoid(output)
        return output.squeeze(-1)