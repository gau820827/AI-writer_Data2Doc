"""Evaluate the model."""
import torch
from dataprepare import loaddata, data2index
from train import evaluate
from model import AttnDecoderRNN, EncoderRNN, EncoderLIN, docEmbedding, EncoderBiLSTM
from settings import file_loc
from util import load_model

import json
import sys, os, configparser
import argparse

config = configparser.ConfigParser()

train_data, train_lang = loaddata(file_loc, 'train')

embedding_size = 600
langs = train_lang
emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                   langs['rm'].n_words, embedding_size)
emb.init_weights()

encoder_src = '../model/boost_LIN/R2_ALL_LIN_encoder_9000'
decoder_src = '../model/boost_LIN/R2_ALL_LIN_decoder_9000'

if 'RNN' in encoder_src:
    encoder = EncoderRNN(embedding_size, emb)
elif 'LSTM' in encoder_src:
    encoder = EncoderBiLSTM(embedding_size, emb)
else:
    encoder = EncoderLIN(embedding_size, emb)

decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)

encoder = load_model(encoder, encoder_src)
decoder = load_model(decoder, decoder_src)

valid_data, _ = loaddata(file_loc, 'valid')
data_length = len(valid_data)
valid_data = data2index(valid_data, train_lang)
evaluate(encoder, decoder, valid_data,
                           train_lang['summary'], embedding_size,
                           iter_time=2, verbose=True)
