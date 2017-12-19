"""Some useful utilizations. Borrowed from Pytorch Tutorial."""
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def showAttention(inputs, outputs, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    score = attentions.numpy()
    for i, text in enumerate(outputs):
        max_score = ['N\A', 0]
        for j, triplet in enumerate(inputs):
            if score[i, j] > max_score[1]:
                max_score = [triplet, score[i, j]]
        print('{} <-> {} = {}'.format(text, max_score[0], max_score[1]))
        if text == '.':
            print('')
