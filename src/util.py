"""Some useful utilizations. Borrowed from Pytorch Tutorial."""
import time
import math
import numpy as np


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
