"""Evaluate the model."""
import torch
from dataprepare import loaddata, data2index
from train import evaluate
from model import AttnDecoderRNN, EncoderRNN, EncoderLIN, docEmbedding
from settings import file_loc
from util import load_model
import json
# Load train_lang
train_data, train_lang = loaddata(file_loc, 'train')

# Load the model
# Initialize the model
embedding_size = 600
langs = train_lang

# import pdb; pdb.set_trace()
emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                   langs['rm'].n_words, embedding_size)
emb.init_weights()


encoder = EncoderRNN(embedding_size, emb)
# encoder = EncoderLIN(embedding_size, emb)
decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)

encoder_src = './model/ALL_RNN_encoder_10000'
decoder_src = './model/ALL_RNN_decoder_10000'

encoder = load_model(encoder, encoder_src)
decoder = load_model(decoder, decoder_src)

# For evaluation
valuation_data = None
with open("../boxscore-data/rotowire/valid.json") as f:
    valuation_data = json.load(f)
assert valuation_data != None

valid_data, _ = loaddata(file_loc, 'valid')
import pdb; pdb.set_trace()
valid_data = data2index(valid_data, train_lang)

# import pdb; pdb.set_trace()
text_generator = evaluate(encoder, decoder, valid_data, train_lang['summary'], embedding_size, verbose=False)

for idx, summary in enumerate(text_generator):

    gold_summary = valuation_data[idx]['summary']
    print('generated summary: \n')
    print(' '.join(summary))
    print('Gold summary: \n')
    print(' '.join(gold_summary))
    break
