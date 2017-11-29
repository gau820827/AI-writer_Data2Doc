"""This is core training part, containing different models."""
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from preprocessing import readfile, data_iter
from settings import file_loc, use_cuda, MAX_LENGTH


# TODO: 1. Finish Evaluation part
#       2. Extend the model


class Lang:
    """A class to summarize encoding information.

    This class will build three dicts:
    word2index, word2count, and index2word for
    embedding information. Once a set of data is
    encoded, we can transform it to corrsponding
    indexing use the word2index, and map it back
    using index2word.

    Attributes:
        word2index: A dict mapping word to index.
        word2count: A dict mapping word to counts in the corpus.
        index2word: A dict mapping index to word.

    """

    def __init__(self, name):
        """Init Lang with a name."""
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>"}
        self.n_words = 3  # Count SOS and EOS

    def addword(self, word):
        """Add a word to the dict."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# embedding(r.t) embedding(r.e) embedding(r.m)
# embedding_dim = 600
# Linear([r.t, r.e, r.m], embedding_dim)

def readLang(data_set):
    """The function to wrap up a data_set.

    This funtion will wrap up a extracted data_set
    into word2index inforamtion. The data set should
    be a list of tuples containing ([triplets], summarize).

    Args:
        data_set: A list of tuples containig 2 items.

    Returns:
        4 Langs for (r.t, r.e, r.m, 'summary')
        (ex. AST, 'Al Hortford', 10, 'summary')

    """
    rt = Lang('rt')
    re = Lang('re')
    rm = Lang('rm')
    summarize = Lang('summarize')

    for v in data_set:
        for triplet in v[0]:
            rt.addword(triplet[0])
            re.addword(triplet[1])
            rm.addword(triplet[2])
        for word in v[1]:
            summarize.addword(word)

    return rt, re, rm, summarize


def preparedata(data_dir):
    """Prepare the data for training."""

    train_set = readfile(data_dir + 'train.json')
    valid_set = readfile(data_dir + 'valid.json')
    test_set = readfile(data_dir + 'test.json')

    rt_train, re_train, rm_train, sum_train = readLang(train_set)

    print("Read %s box score summary" % len(train_set))
    print("Embedding size of (r.t, r.e, r.m) and summary:")
    print("({}, {}, {}), {}".format(rt_train.n_words, re_train.n_words, rm_train.n_words, sum_train.n_words))

    # Extend the dataset with indexing
    for i in range(len(train_set)):
        idx_triplets = []
        for triplet in train_set[i][0]:
            idx_triplet = [None, None, None]
            idx_triplet[0] = rt_train.word2index[triplet[0]]
            idx_triplet[1] = re_train.word2index[triplet[1]]
            idx_triplet[2] = rm_train.word2index[triplet[2]]
            idx_triplets.append(tuple(idx_triplet))

        idx_summary = []
        for word in train_set[i][1]:
            idx_summary.append(sum_train.word2index[word])
        idx_summary.append(1)   # Append 'EOS' at the end

        train_set[i].append([idx_triplets] + [idx_summary])

    return train_set, rt_train, re_train, rm_train, sum_train


def get_batch(batch):
    """Get the batch into training format.

    Arg:
        batch: The iterator of the dataset

    Returns:
        batch_data: The origin processed data
                    (i.e batch_size * (triples, summary))
        batch_idx_data: The indexing-processed data
                        (e.g batch_size * (r.t, r.e, r.m, summary))

    """
    batch_data = []
    batch_idx_data = [[], [], [], []]
    max_summary_length = 0
    for d in batch:
        idx_data = [[], [], []]  # for each triplet
        batch_data.append(d[:2])
        for triplets in d[2][0]:
            for idt, t in enumerate(triplets):
                idx_data[idt].append(t)

        for idb, b in enumerate(idx_data):
            batch_idx_data[idb].append(b)

        # Calculate maximum length of the summary
        max_summary_length = max(max_summary_length, len(d[2][1]))

        batch_idx_data[3].append(d[2][1])

    # Add paddings for summary
    for i in range(len(batch_idx_data[3])):
        paddings = [2 for k in range(len(batch_idx_data[3][i]) - max_summary_length)]
        batch_idx_data[3][i] += paddings

    return batch_data, batch_idx_data


class docEmbedding(nn.Module):
    """The class for embedding records.

    This class is for embedding the docvec (r.t, r.e, r.m)
    into a high dimension space. A MLP with RELU will be applied
    on the concatenation of the embeddings.

    Attributes:
        embedding1: embedding for r.t
        embedding2: embedding for r.e
        embedding3: embedding for r.m
        linear: A linear layer mapping [r.t, r.e, r.m] back to one space

    """
    def __init__(self, rt_size, re_size, rm_size, embedding_dim):
        super(docEmbedding, self).__init__()
        self.embedding1 = nn.Embedding(rt_size, embedding_dim)
        self.embedding2 = nn.Embedding(re_size, embedding_dim)
        self.embedding3 = nn.Embedding(rm_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, rt, re, rm):
        emb_rt = self.embedding1(rt)
        emb_re = self.embedding2(re)
        emb_rm = self.embedding3(rm)

        emb_all = torch.cat([emb_rt, emb_re, emb_rm], dim=1)
        output = self.linear(emb_all)
        output = F.relu(output)
        return output

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear]
        em_layer = [self.embedding1, self.embedding2, self.embedding3]

        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_layer, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = embedding_layer
        self.gru = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, rt, re, rm, hidden):
        embedded = self.embedding(rt, re, rm)
        output = embedded

        for i in range(self.n_layers):
            output = self.gru(output, hidden)

        return output

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), dim=1)))
        attn_weights = attn_weights.unsqueeze(1)

        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        attn_applied = attn_applied.squeeze(1)

        output = torch.cat((embedded, attn_applied), dim=1)
        output = self.attn_combine(output)

        for i in range(self.n_layers):
            output = F.relu(output)
            output = self.gru(output, hidden)

        output_hidden = output
        output = F.log_softmax(self.out(output))
        return output, output_hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


def addpaddings(summary):
    """A helper function to add paddings to summary.

    Args:
        summary: A list (batch_size) of indexing summarizes.

    Returns:
        A list (batch_size) with padding summarizes.
    """
    max_length = len(max(summary, key=len))
    for i in range(len(summary)):
        summary[i] += [2 for i in range(max_length - len(summary[i]))]
    return summary


def train():
    # Temp parameter
    embedding_size = 600
    learning_rate = 0.1
    iter_time = 10

    train_set, rt_train, re_train, rm_train, sum_train = preparedata(file_loc)
    train_iter = data_iter(train_set)

    # Initialize the model
    emb = docEmbedding(rt_train.n_words, re_train.n_words,
                       rm_train.n_words, embedding_size)
    emb.init_weights()

    encoder = EncoderRNN(embedding_size, emb)
    decoder = AttnDecoderRNN(embedding_size, sum_train.n_words)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iteration in range(iter_time):

        # Get data
        data, idx_data = get_batch(next(train_iter))
        rt, re, rm, summary = idx_data

        # Add paddings
        rt = addpaddings(rt)
        re = addpaddings(re)
        rm = addpaddings(rm)
        summary = addpaddings(summary)

        # For Encoding
        rt = Variable(torch.LongTensor(rt))
        re = Variable(torch.LongTensor(re))
        rm = Variable(torch.LongTensor(rm))

        # For Decoding
        summary = Variable(torch.LongTensor(summary))

        # Zero the gradient
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        batch_length = rt.size()[0]
        input_length = rt.size()[1]
        target_length = summary.size()[1]

        encoder_outputs = Variable(torch.zeros(batch_length, MAX_LENGTH, embedding_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        encoder_hidden = encoder.initHidden(batch_length)

        loss = 0

        # Encoding
        for ei in range(input_length):
            encoder_hidden = encoder(rt[:, ei], re[:, ei], rm[:, ei], encoder_hidden)

            # Store memory information
            encoder_outputs[:, ei] = encoder_hidden

        decoder_hidden = encoder_hidden

        decoder_input = Variable(torch.LongTensor(batch_length).zero_())
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        teacher_forcing_ratio = 0.001
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_hidden,
                    encoder_outputs)

                loss += criterion(decoder_output, summary[:, di])
                decoder_input = summary[:, di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_hidden,
                    encoder_outputs)

                topv, topi = decoder_output.data.topk(1)

                decoder_input = Variable(topi.squeeze(1))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output, summary[:, di])
                # if ni == '<EOS>':
                #     break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        print("iter {}, loss = {}".format(iteration, loss))

if __name__ == '__main__':
    train()
