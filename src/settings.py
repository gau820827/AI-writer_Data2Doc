"""The file contains name settings."""
import torch
file_loc = '../boxscore-data/rotowire/'
use_cuda = torch.cuda.is_available()
MAX_LENGTH = 630
LAYER_DEPTH = 2
MAX_SENTENCE = None

EMBEDDING_SIZE = 600
LR = 0.01
ITER_TIME = 10000
BATCH_SIZE = 8
