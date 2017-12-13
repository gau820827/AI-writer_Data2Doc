
import numpy as np
import random
import argparse
import h5py


class Prepare_data:
    """
    Read the h5 file and get batches
    """
    def __init__(self, args, datatype='tr'):
        self.h5fi = h5py.File(args.input_path, 'r')
        self.sen = self.h5fi[datatype + 'sents'].value
        self.len = self.h5fi[datatype + 'lens'].value
        self.numdists= self.h5fi[datatype + 'numdists'].value
        self.entdists = self._shift_nagative(self.h5fi[datatype + 'entdists'].value)
        self.numdists = self._shift_nagative(self.h5fi[datatype + 'numdists'].value)

        self.labels = self.h5fi[datatype + 'labels'].value
        self.labelnum = np.asarray(list(map(lambda x : x[-1], self.labels)))

        self.label_pad = np.amax(self.labels)
        self.word_pad = np.amax(self.sen) + 1
        self.ent_dist_pad = np.amax(self.entdists) + 1
        self.num_dist_pad = np.amax(self.numdists) + 1

    def __def__ (self):
        self.h5fi.close()

    def get_idx_range(self):
        return  self.label_pad, self.word_pad, \
                self.ent_dist_pad, self.num_dist_pad

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
            sen_batch = [self.sen[idx] for idx in batch_indices]
            ent_dist_batch = [self.entdists[idx] for idx in batch_indices]
            num_dist_batch = [self.numdists[idx] for idx in batch_indices]
            label_batch = [self.labels[idx] for idx in batch_indices]
            label_num_batch = [self.labelnum[idx] for idx in batch_indices]
            start += batch_size
            yield sen_batch, ent_dist_batch, num_dist_batch, label_batch, label_num_batch
        # import pdb; pdb.set_trace()
        # print('hello')

    def _shift_nagative(self, x):
        min_label = np.amin(x)
        for row in x:
            row += - min_label + 1
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data prepare class')
    parser.add_argument('-input_path', type=str, default="roto-ie.h5",
                        help="h5df file path")
    args = parser.parse_args()
    data = Prepare_data(args)
    aaa, bbb, ccc, ddd, eee = next(data.get_batch())
    print('hello')
