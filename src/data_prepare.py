import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
import argparse
import h5py

from cnn_model import Size_info

class Prepare_data:
    """
    Read the h5 file and get batches
    """
    def __init__(self, filepath, datatype='tr'):
        self.datatype = datatype
        self.h5fi = h5py.File(filepath, 'r')
        self.sen = self.h5fi[datatype + 'sents'].value
        # the length of the sentences
        self.len = self.h5fi[datatype + 'lens'].value
        self.entdists = self._shift_nagative(self.h5fi[datatype + 'entdists'].value)
        self.numdists = self._shift_nagative(self.h5fi[datatype + 'numdists'].value)
        # to do cut the vector by the length
        # add padding
        # always using numpy arrayt
        self.labels = self.h5fi[datatype + 'labels'].value
        self.labelnum = list(map(lambda x : x[-1], self.labels))

        self.label_pad = np.amax(self.labels)
        self.word_pad = np.amax(self.sen) + 1
        self.ent_dist_pad = np.amax(self.entdists) + 1
        self.num_dist_pad = np.amax(self.numdists) + 1
        # import pdb; pdb.set_trace()

        self.sen = self._add_padding(self.sen, self.word_pad)
        self.numdists = self._add_padding(self.numdists, self.num_dist_pad)
        self.entdists = self._add_padding(self.entdists, self.ent_dist_pad)
        self.labels = self._add_padding(self.labels, self.label_pad)

    def __def__ (self):
        self.h5fi.close()

    def get_size_info(self):
        size_info = Size_info(label_size=self.label_pad, word_embed_size=self.word_pad, \
                  entpos_size=self.ent_dist_pad, numpos_size=self.num_dist_pad, \
                  max_len=np.amax(self.len))
        return size_info

    def _add_padding(self, x, padding):
        x[x == -1] = padding;
        # import pdb; pdb.set_trace()
        return x



    def get_batch(self, batch_size=32):
        data_size = len(self.sen)
        order = list(range(data_size))
        random.shuffle(order)
        start = 0
        while True:
            if start > data_size - batch_size:
                start = 0
                random.shuffle(order)
            batch_indices = order[start : start + batch_size]
            lengths = np.array([self.len[idx] for idx in batch_indices ])
            max_length = np.amax(lengths)
            sen_batch = np.array([self.sen[idx][:max_length]for idx in batch_indices]) - 1
            # import pdb; pdb.set_trace()
            ent_dist_batch = np.array([self.entdists[idx][:max_length] for idx in batch_indices]) - 1
            num_dist_batch = np.array([self.numdists[idx][:max_length] for idx in batch_indices]) - 1
            # do somthing else if the datatype is test
            label_num_batch = np.array([self.labelnum[idx] for idx in batch_indices])
            # label_batch = [self.labels[idx][:self.labelnum[idx]] for idx in batch_indices]
            label_batch = np.array([self.labels[idx][0] for idx in batch_indices]) - 1
            start += batch_size
            # import pdb; pdb.set_trace()
            yield sen_batch, ent_dist_batch, num_dist_batch, label_batch, label_num_batch
        # import pdb; pdb.set_trace()
        # print('hello')

    def _shift_nagative(self, x):
        min_label = np.amin(x)
        for row in x:
            row += - min_label + 1
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main for training the CNN extraction system')
    parser.add_argument('-input_path', type=str, default="roto-ie.h5",
                        help="h5df file path")
    parser.add_argument('-output_fi', type=str, default="",
                        help="desired path to output file")
    parser.add_argument('-label', type=str, default="roto-ie.labels",
                        help="file containing label to index")
    parser.add_argument('-word_dict', type=str, default='roto-ie.dict',
                        help="file containing word to index")
    parser.add_argument('-test', action='store_true', help='use test data')
    parser.add_argument('-save', action='store_true', help='save the model or not')

    args = parser.parse_args()
    data = Prepare_data(args.input_path)
    aaa, bbb, ccc, ddd, eee = next(data.get_batch())
    print('hello')
