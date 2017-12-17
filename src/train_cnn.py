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
from settings import use_cuda

config = configparser.ConfigParser()
from time import gmtime, strftime

def get_batch_loss(model, optimizer, criterion, sen_batch,
                ent_dist_batch, num_dist_batch, label_batch):
     optimizer.zero_grad()
     model.train()
    #  import pdb; pdb.set_trace()
     pred = model(sen_batch, ent_dist_batch, num_dist_batch)
     loss = criterion(pred, label_batch)
     loss.backward()
     optimizer.step()
     return loss

def train(training_data):
    num_iteration = config.getint('train', 'iteration')
    batch_size = config.getint('train', 'batch_size')
    learning_rate = config.getfloat('train', 'learning_rate')
    save_file = config.get('train', 'save_file')
    start_time = time.time()
    model = Conv_relation_extractor(config, training_data.get_size_info())
    if use_cuda:
        model.cuda()
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    for epoch in range(1, num_iteration + 1):
        print("epoch " + str(epoch) + " has begun.")
        data_iterator = training_data.get_batch()
        print_count = 0
        for sen_batch, ent_dist_batch, num_dist_batch,\
            label_batch, label_num_batch in data_iterator:
            # import pdb; pdb.set_trace()
            sen_batch = Variable(torch.from_numpy(sen_batch))
            ent_dist_batch = Variable(torch.from_numpy(ent_dist_batch))
            num_dist_batch = Variable(torch.from_numpy(num_dist_batch))
            # labels
            label_batch =  Variable(torch.from_numpy(label_batch))
            if use_cuda:
                sen_batch, ent_dist_batch = sen_batch.cuda(), ent_dist_batch.cuda()
                num_dist_batch, label_batch = num_dist_batch.cuda(), label_batch.cuda()
            loss = get_batch_loss(model, optimizer, criterion, sen_batch, ent_dist_batch,
                           num_dist_batch, label_batch)

            print_count += 1
            if print_count % 100 == 0:
                print("epoch {}, batchs {} done.\n".format(epoch, print_count))
                print(loss)
        current_time = strftime("%Y-%m-%d-%H_%M_%S", gmtime())
        torch.save(model.state_dict(), "{}_{}_{}.model".format(save_file, epoch, current_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main for training the CNN extraction system')
    parser.add_argument('-input_path', type=str, default="mini_test.h5",
                        help="h5df file path")
    parser.add_argument('-config', type=str, default="config.cfg",
                        help="path to config file.")
    # parser.add_argument('-label', type=str, default="roto-ie.labels",
    #                     help="file containing label to index")
    # parser.add_argument('-word_dict', type=str, default='roto-ie.dict',
    #                     help="file containing word to index")
    # parser.add_argument('-test', action='store_true', help='use test data')
    # parser.add_argument('-notsave', action='store_true', help='not save the model')
    args = parser.parse_args()
    config.read(args.config)
    training_data = Prepare_data(args.input_path)
    train(training_data)
