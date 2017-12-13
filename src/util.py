"""Some useful utilizations. Borrowed from Pytorch Tutorial."""
import time
import math
import numpy as np
import torch


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def gettime(start):
    now = time.time()
    return asMinutes(now - start)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def load_model(model, model_src, mode='eval'):
    state_dict = torch.load(model_src, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    if mode == 'eval':
        model.eval()
    else:
        model.train()

    return model
