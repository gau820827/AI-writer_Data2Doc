"""Evaluate the model."""
import torch
from dataprepare import loaddata, data2index
from train import evaluate
from model import AttnDecoderRNN, EncoderRNN, EncoderLIN, docEmbedding
from settings import file_loc
from util import load_model
import json


train_data, train_lang = loaddata(file_loc, 'train')
embedding_size = 600
langs = train_lang
emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                   langs['rm'].n_words, embedding_size)
emb.init_weights()
# encoder = EncoderRNN(embedding_size, emb)
# # encoder = EncoderLIN(embedding_size, emb)
# decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)

def generate_text():
    encoder_src = './model/ALL_RNN_encoder_10000'
    decoder_src = './model/ALL_RNN_decoder_10000'
    # encoder_src = model[0]
    # decoder_src = model[1]
    encoder = EncoderRNN(embedding_size, emb)
    # encoder = EncoderLIN(embedding_size, emb)
    decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)
    encoder = load_model(encoder, encoder_src)
    decoder = load_model(decoder, decoder_src)
    valuation_data = None
    with open("./mini.json") as f:
        valuation_data = json.load(f)
    assert valuation_data != None

    valid_data, _ = loaddata('./', 'mini')
    data_length = len(valid_data)
    valid_data = data2index(valid_data, train_lang)
    text_generator = evaluate(encoder, decoder, valid_data,
                            train_lang['summary'], embedding_size,
                            iter_time = data_length , verbose=False)
    print('The text generation begin\n', flush=True)
    with open('ALL_RNN_encoder_mini.text', 'w') as f:
        for line in text_generator:
            print('Summery generated')
            f.write(line + '\n')





if __name__ == "__main__":
    generate_text()


# for idx, summary in enumerate(text_generator):
    # valuation_data[idx]['summary'] = summary
    # gold_summary = valuation_data[idx]['summary']
    # print('generated summary: \n')
    # print(' '.join(summary))
    # print('Gold summary: \n')
    # print(' '.join(gold_summary))
    # break
# to_test = json.dump(valuation_data)

# with open('to_test.json', 'w') as f:
    # f.write(to_test)
