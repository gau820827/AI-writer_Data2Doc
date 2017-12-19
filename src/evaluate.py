"""Evaluate the model."""
import torch
from dataprepare import loaddata, data2index
from train import evaluate
from model import AttnDecoderRNN, EncoderRNN, EncoderLIN, docEmbedding
from settings import file_loc
from util import load_model

# Load train_lang
train_data, train_lang = loaddata(file_loc, 'train')

# Load the model
# Initialize the model
embedding_size = 600
langs = train_lang
emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                   langs['rm'].n_words, embedding_size)
emb.init_weights()

# encoder = EncoderRNN(embedding_size, emb)
encoder = EncoderLIN(embedding_size, emb)
decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)

encoder_src = '../model/boost_LIN/R2_ALL_LIN_encoder_10000'
decoder_src = '../model/boost_LIN/R2_ALL_LIN_decoder_10000'

encoder = load_model(encoder, encoder_src)
decoder = load_model(decoder, decoder_src)

# For evaluation
valid_data, _ = loaddata(file_loc, 'valid')
valid_data_idx = data2index(valid_data, train_lang)
evaluate(encoder, decoder, valid_data_idx, train_lang['summary'], embedding_size, showAtten=True)
