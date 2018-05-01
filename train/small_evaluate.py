"""Evaluate the model."""
from dataprepare import loaddata, data2index
from train import evaluate, model_initialization
from model import AttnDecoderRNN, EncoderRNN, EncoderLIN, docEmbedding, EncoderBiLSTM
from settings import file_loc, ENCODER_STYLE, DECODER_STYLE, USE_MODEL
from settings import EMBEDDING_SIZE, LR
from util import load_model


train_data, train_lang = loaddata(file_loc, 'train')

langs = train_lang

encoder, decoder, _, _ = model_initialization(ENCODER_STYLE, DECODER_STYLE, langs, EMBEDDING_SIZE, LR, USE_MODEL)

valid_data, _ = loaddata(file_loc, 'valid')
data_length = len(valid_data)
valid_data = data2index(valid_data, train_lang)
text_generator = evaluate(encoder, decoder, valid_data,
                          train_lang['summary'], EMBEDDING_SIZE,
                          encoder_style=ENCODER_STYLE, iter_time=2,
                          beam_size=1, verbose=False)

# Generate Text
for idx, text in enumerate(text_generator):
    print('Generate Summary {}:\n{}'.format(idx + 1, text))
