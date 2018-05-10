"""The file contains name settings."""
import torch
file_loc = '../boxscore-data/rotowire/'
use_cuda = torch.cuda.is_available()
# use_cuda = False

MAX_LENGTH = 800
# USE_MODEL = None
# Model_name = None
# Model_name = 'pl_copy_test'
# iterNum = 120
# if Model_name is not None:
#     USE_MODEL = ['./models/'+Model_name + '_' + s + '_' + str(iterNum) for s in ['encoder', 'decoder', 'optim']]
# USE_MODEL = ['./models/clipped_encoder_25440', './models/clipped_decoder_25440']
# USE_MODEL = ['./models/long3_encoder_36040',
#              './models/long3_decoder_36040',
# './models/long3_optim_36040']

# Parameter for training
MAX_SENTENCES = None
MAX_TRAIN_NUM = 20

# PRETRAIN is the model name that you want read
# The naming convention is 'PRETRAIN_iterNum'
PRETRAIN = 'hi'
# PRETRAIN = None
# iterNum = None
iterNum = 360

# Default hyper-parameters for training
EMBEDDING_SIZE = 600
LR = 0.01  # Adagrad
# LR = 0.003  # Adam
EPOCH_TIME = 100
BATCH_SIZE = 1
GRAD_CLIP = 3
LAYER_DEPTH = 1

# Default parameters for display
GET_LOSS = 1
SAVE_MODEL = 1

# Choose models

# ENCODER_STYLE = 'LIN'
# ENCODER_STYLE = 'RNN'
# ENCODER_STYLE = 'BiLSTM'
# DECODER_STYLE = 'RNN'

ENCODER_STYLE = 'HierarchicalRNN'
DECODER_STYLE = 'HierarchicalRNN'
OUTPUT_FILE = 'hi'
COPY_PLAYER = True
TOCOPY = False

# DATA PREPROCESSING
MAX_PLAYERS = 31  # information taken from rotowire
PLAYER_PADDINGS = ['<PAD' + str(i) + '>' for i in range(0, MAX_PLAYERS)]
