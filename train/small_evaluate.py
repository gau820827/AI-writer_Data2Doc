"""Evaluate the model."""
from dataprepare import loaddata, data2index
from train import evaluate
from model import AttnDecoderRNN, EncoderRNN, EncoderLIN, docEmbedding, EncoderBiLSTM
from settings import file_loc
from util import load_model


train_data, train_lang = loaddata(file_loc, 'train')

embedding_size = 600
langs = train_lang
emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                   langs['rm'].n_words, embedding_size)
emb.init_weights()

encoder_src = '../model/demo/ALL_BiLSTM_encoder_10000'
decoder_src = '../model/demo/ALL_BiLSTM_decoder_10000'

encoder_style = None

if 'RNN' in encoder_src:
    encoder = EncoderRNN(embedding_size, emb)
    encoder_style = 'RNN'
elif 'LSTM' in encoder_src:
    encoder = EncoderBiLSTM(embedding_size, emb)
    encoder_style = 'BiLSTM'
else:
    encoder = EncoderLIN(embedding_size, emb)
    encoder_style = 'LIN'

decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)

encoder = load_model(encoder, encoder_src)
decoder = load_model(decoder, decoder_src)

valid_data, _ = loaddata(file_loc, 'valid')
data_length = len(valid_data)
valid_data = data2index(valid_data, train_lang)
text_generator = evaluate(encoder, decoder, valid_data,
                          train_lang['summary'], embedding_size,
                          encoder_style=encoder_style, iter_time=2,
                          beam_size=1, verbose=False)

# Generate Text
for idx, text in enumerate(text_generator):
    print('Generate Summary {}:\n{}'.format(idx + 1, text))
