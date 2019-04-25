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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import my_utils

from configs import ParseParams
from myModels import lilyNN

class GenerateTest(Dataset):
    def __init__(self, data_size, seed, args):
        super(GenerateTest, self).__init__()
        self.num_sample = data_size
        self.seed = seed
        if args.data_dir == '':
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            input = torch.rand((self.num_sample, args.feature_num, args.input_size))
            output = torch.rand((self.num_sample))
        else:
            pass
        self.input = input
        self.output = output
    def __len__(self):
        return self.num_sample
    def __getitem__(self, batch_idx):
        return (self.input[batch_idx], self.output[batch_idx])      


def train(NNmodel, train_data, valid_data, args):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_optim = optim.Adam(NNmodel.parameters(), lr=args.lr)
    train_loader = DataLoader(train_data, args.batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, args.batch_size, False, num_workers=0)

    for epoch in range(args.num_epoch):
        NNmodel.train()
        for batch_idx, batch in enumerate(train_loader):
            input, output = batch

            pred_output = NNmodel(input)

            loss = ((pred_output-output) ** 2).mean()
            model_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(NNmodel.parameters(), args.max_grad_norm)
            model_optim.step()
        
            if batch_idx % 50 ==0:    
                print('#{:d} loss: {:.2f}'.format(
                    batch_idx, loss))

if __name__ == '__main__':
    args = ParseParams()
    NNmodel = lilyNN(args.feature_num, args.hidden_size, args.input_size)

    train_data = GenerateTest(int(1e5), args.seed, args)
    valid_data = GenerateTest(int(1e3), args.seed+1, args)
    test_data = GenerateTest(int(1e3), args.seed+2, args)

    if not args.test:
        train(NNmodel, train_data, valid_data, args)