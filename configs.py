import os
import time
import pickle
import argparse
import datetime
import numpy as np
import torch


def ParseParams(mode='ML'):
    parser = argparse.ArgumentParser(
        description='NN for similarity')

    # task
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', help='test or train', default=False)
    parser.add_argument('--task', dest='task', default='order_similarity')
    parser.add_argument('--input_size', dest='input_size', default=2)
    parser.add_argument('--feature_num', dest='feature_num', default=10)
    parser.add_argument('--hidden_size', dest='hidden_size', default=128)

    # training hyper-parameters
    parser.add_argument('--num_epoch', dest='num_epoch', default=5)
    parser.add_argument('--batch_size', dest='batch_size', default=128)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_grad_norm', default=2, type=float)
    

    # dirs
    parser.add_argument('--data_dir', dest='data_dir', default='')
    parser.add_argument('--log_dir', dest='log_dir', default='logs')
    parser.add_argument('--model_dir', dest='model_dir', default='model')
    parser.add_argument('--check_point', dest='check_point', default='')

    args, unknown = parser.parse_known_args()
    ''' modify here!!!'''
    args.test = False

    ''' modify here!!!'''

    return args