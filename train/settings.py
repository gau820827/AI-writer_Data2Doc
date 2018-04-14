"""The file contains name settings."""
import torch
file_loc = '../boxscore-data/rotowire/'
use_cuda = torch.cuda.is_available()
MAX_LENGTH = 664
LAYER_DEPTH = 2
MAX_SENTENCES = None
#USE_MODEL = None
USE_MODEL = ['./models/clipped_encoder_25440', './models/clipped_decoder_25440']

# Parameter for training
EMBEDDING_SIZE = 600
LR = 0.01
ITER_TIME = 60
BATCH_SIZE = 8

# Parameter for display
GET_LOSS = 30
SAVE_MODEL = 10
# ENCODER_STYLE = 'BiLSTM'
ENCODER_STYLE = 'RNN'
OUTPUT_FILE = 'clipped2'
