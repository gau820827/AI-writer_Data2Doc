"""The file contains name settings."""
import torch
file_loc = '../boxscore-data/rotowire/'
use_cuda = torch.cuda.is_available()
MAX_LENGTH = 683
#MAX_LENGTH = 8
LAYER_DEPTH = 2
#MAX_SENTENCES = None
MAX_SENTENCES = 5
USE_MODEL = None
#USE_MODEL = ['./models/clipped_encoder_25440', './models/clipped_decoder_25440']
# USE_MODEL = ['./models/long3_encoder_36040',
#              './models/long3_decoder_36040',
             # './models/long3_optim_36040']
# Parameter for training
EMBEDDING_SIZE = 600
LR = 0.001
ITER_TIME = 220
BATCH_SIZE = 8
GRAD_CLIP = 5

# Parameter for display
GET_LOSS = 1
SAVE_MODEL = 1
ENCODER_STYLE = 'BiLSTM'
#ENCODER_STYLE = 'RNN'
#DECODER_STYLE = 'HierarchicalRNN'
DECODER_STYLE = 'RNN'
OUTPUT_FILE = 'test_b8_max5'
USE_MODEL = None
TOCOPY = True

# DATA PREPROCESSING
""" Ken added """
MAX_PLAYERS = 30  # information taken from rotowire
PLAYER_PADDINGS = ['<PAD' + str(i) + '>' for i in range(0, MAX_PLAYERS)]
