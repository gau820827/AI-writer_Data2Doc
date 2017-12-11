"""This is the file for main model."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from settings import use_cuda, MAX_LENGTH, LAYER_DEPTH


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


class EncoderLIN(nn.Module):
    """This is the linear encoder for the box score.

    From the origin paper, they use a linear encoder instead of standard
    sequential RNN style encoder. The encoder will mean pool over the entities
    and then linearly transform the concatenation of these pooled entity representations
    to initialize the decoder.

    """
    def __init__(self, hidden_size, embedding_layer):
        super(EncoderLIN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding_layer
        self.avgpool = nn.AvgPool1d(3, stride=2, padding=1)

    def forward(self, rt, re, rm, hidden):
        embedded = self.embedding(rt, re, rm)
        output = torch.cat((embedded, hidden), dim=1)
        output = self.avgpool(output.view(-1, 1, 2 * self.hidden_size))
        return output.squeeze(1)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_layer, n_layers=LAYER_DEPTH):
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
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH, n_layers=LAYER_DEPTH, dropout_p=0.5):
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
