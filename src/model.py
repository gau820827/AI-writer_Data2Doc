"""This is the file for main model."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from settings import use_cuda, MAX_LENGTH


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
