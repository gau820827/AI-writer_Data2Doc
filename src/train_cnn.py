import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from util import gettime
from cnn_model import *
from data_prepare import Prepare_data
import sys, configparser
import argparse
from settings from use_cuda

config = configparser.ConfigParser()

def train(training_data):
    model = Conv_relation_extractor(config, training_data.get_size_info())
    num_iteration = config.getint('train', 'iteration')
    batch_size = config.getint('train', 'batch_size')
    for iteration in range(1, num_iteration + 1):
        
    pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main for training the CNN extraction system')
    parser.add_argument('-input_path', type=str, default="roto-ie.h5",
                        help="h5df file path")
    parser.add_argument('-config', type=str, default="config.cfg",
                        help="path to config file.")
    parser.add_argument('-label', type=str, default="roto-ie.labels",
                        help="file containing label to index")
    parser.add_argument('-word_dict', type=str, default='roto-ie.dict',
                        help="file containing word to index")
    parser.add_argument('-test', action='store_true', help='use test data')
    parser.add_argument('-notsave', action='store_true', help='not save the model')
    args = parser.parse_args()
    config.read(args.config)
    training_data = Prepare_data(args.input_path)
    train(training_data)
