"""Some useful utilizations. Borrowed from Pytorch Tutorial."""
import time
import math
import heapq

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


def show_attention(inputs, outputs, attentions):
    """The function to show attention scores.

    Args:
        inputs: A list of tuples, indicating the input triplets.
        outputs: A list of strings, indicating the output texts.
        attentions: A matrix of attention scores.

    """
    score = attentions.numpy()
    for i, text in enumerate(outputs):
        max_score = ['N\A', 0]
        for j, triplet in enumerate(inputs):
            if score[i, j] > max_score[1]:
                max_score = [triplet, score[i, j]]
        print('{} <-> {} = {}'.format(text, max_score[0], max_score[1]))
        if text == '.':
            print('')


def show_triplets(triplets):
    """The function to show input triplets.

    Args:
        triplets: A list of tuples, indicating the input triplets.

    """
    for triplet in triplets:
        print(triplet, end=',')
        if triplet[2] == '<EOB>':
            print('\n==============')
    return


class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]
