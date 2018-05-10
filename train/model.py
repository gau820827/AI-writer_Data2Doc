"""This is the file for main model."""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from settings import use_cuda, MAX_LENGTH, LAYER_DEPTH, TOCOPY


class Seq2Seq(object):
    def __init__(self, encoder, decoder, train_func, criterion, embedding_size, langs):
        self.encoder = encoder
        self.decoder = decoder
        self.train_func = train_func
        self.criterion = criterion
        self.embedding_size = embedding_size
        self.langs = langs

    def seq_train(self, rt, re, rm, summary):
        """The function to calculate the loss on one batch."""
        return self.train_func(rt, re, rm, summary,
                               self.encoder, self.decoder,
                               self.criterion, self.embedding_size, self.langs)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()


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

        emb_all = torch.cat([emb_rt, emb_re, emb_rm], dim=len(rt.size()))
        output = F.relu(self.linear(emb_all))
        # output = self.linear(emb_all)
        return output

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear]
        em_layer = [self.embedding1, self.embedding2, self.embedding3]

        for layer in lin_layers + em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)


class HierarchicalLIN(nn.Module):
    def __init__(self, hidden_size, local_embed):
        super(HierarchicalLIN, self).__init__()
        self.LocalEncoder = EncoderLIN(hidden_size, local_embed, level='local')
        self.GlobalEncoder = EncoderLIN(hidden_size, None, level='global')


class EncoderLIN(nn.Module):
    """This is the linear encoder for the box score.

    From the original paper, they use a linear encoder instead of standard
    sequential RNN style encoder. The encoder will mean pool over the entities
    and then linearly transform the concatenation of these pooled entity
    representations to initialize the decoder.

    """

    def __init__(self, hidden_size, embedding_layer, level='plain'):
        """."""
        super(EncoderLIN, self).__init__()
        self.name = 'LIN'
        self.level = level
        self.hidden_size = hidden_size
        if self.level == 'plain' or self.level == 'local':
            self.embedding = embedding_layer
        self.avgpool = nn.AvgPool1d(32)

    def forward(self, inputs, hidden):
        """Dims."""
        # rt (n_batch, seq_len)
        # embedded (n_batch, seq_len, emb_dim)
        # global inp: MAX_BLOCK, batch_length, input_length
        # hiddens (max_length, batch, hidden size)

        if self.level == 'global':
            # AvgPool for each row as R, AvgPool for each
            inp = inputs['local_hidden_states'].permute(2, 1, 0)
            # inp: (seq_len, batch, dimension)
            seq_len = inp.size(0)
            outputs = F.avg_pool1d(inp, 32)
            # 32 is the size of each block
            hidden = F.avg_pool1d(outputs, int(seq_len / 32)).permute(2, 1, 0)
            outputs = outputs.permute(2, 1, 0)
        else:
            # Local and Plain.
            inp = self.embedding(inputs['rt'], inputs['re'], inputs['rm'])
            outputs = inp.permute(1, 0, 2)
            hidden = outputs[-1, :, :]
        return outputs, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        if use_cuda:
            return result.cuda()
        else:
            return result


class HierarchicalRNN(nn.Module):
    def __init__(self, hidden_size, local_embed, n_layers=LAYER_DEPTH):
        super(HierarchicalRNN, self).__init__()
        self.LocalEncoder = EncoderRNN(hidden_size, local_embed,
                                       n_layers=n_layers, level='local')
        self.GlobalEncoder = EncoderRNN(hidden_size, None,
                                        n_layers=n_layers, level='global')


class EncoderRNN(nn.Module):
    """Vanilla encoder using pure gru."""
    def __init__(self, hidden_size, embedding_layer, n_layers=LAYER_DEPTH, level='plain'):
        super(EncoderRNN, self).__init__()
        self.name = 'RNN'
        self.level = level
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if self.level == 'local' or self.level == 'plain':
            self.embedding = embedding_layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers)

    def forward(self, inputs, hidden):
        # emb (n_batch, seq_len, emb_dim)
        # inp (seq_len, batch, emb_dim)
        # gru needs (seq_len, n_batch, emb_dim)
        if self.level == 'global':
            outputs, hidden = self.gru(inputs['local_hidden_states'], hidden)
        else:
            embedded = self.embedding(inputs['rt'], inputs['re'], inputs['rm'])
            inp = embedded.permute(1, 0, 2)
            if self.level == 'plain':
                outputs, hidden = self.gru(inp, hidden)
            else:
                # Local.
                seq_len, batch_size, embed_dim = inp.size()
                outputs = Variable(torch.zeros(seq_len, batch_size, embed_dim))
                outputs = outputs.cuda() if use_cuda else outputs
                for ei in range(seq_len):
                    if ei > 0 and ei % 32 == 0:
                        hidden = self.initHidden(batch_size)
                    seq_i = inp[ei, :, :].unsqueeze(0)
                    # seq_i of size: (1, batch, emb_dim)
                    output, hidden = self.gru(seq_i, hidden)
                    # output of size: (1, batch, emb_dim)
                    outputs[ei, :, :] = output[0, :, :]
        # outputs (seq_len, batch, hidden_size * num_directions)
        # hidden is the at t = seq_len
        return outputs, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size), requires_grad=False)

        if use_cuda:
            return result.cuda()
        else:
            return result


class HierarchicalBiLSTM(nn.Module):
    """"""
    def __init__(self, hidden_size, local_embed, n_layers=LAYER_DEPTH):
        super(HierarchicalBiLSTM, self).__init__()
        self.LocalEncoder = EncoderBiLSTM(hidden_size, local_embed,
                                          n_layers=n_layers, level='local')
        self.GlobalEncoder = EncoderBiLSTM(hidden_size, None,
                                           n_layers=n_layers, level='global')


class EncoderBiLSTM(nn.Module):
    """Vanilla encoder using pure LSTM."""
    def __init__(self, hidden_size, embedding_layer, n_layers=LAYER_DEPTH, level='plain'):
        super(EncoderBiLSTM, self).__init__()
        self.name = 'BiLSTM'
        self.level = level
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if self.level == 'plain' or self.level == 'local':
            self.embedding = embedding_layer
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=n_layers, bidirectional=True)

    def forward(self, inputs, hidden):
        # embedded is of size (n_batch, seq_len, emb_dim)
        # lstm needs: (seq_len, batch, input_size)
        # lstm output: (seq_len, batch, hidden_size * num_directions)
        if self.level == 'global':
            inp = inputs['local_hidden_states']
            outputs, (hn, cn) = self.bilstm(inp, hidden)
            # hn: (num_layers * num_directions, batch, hidden_size):
            return outputs, hn.view(self.n_layers, -1, self.hidden_size)
        else:
            embedded = self.embedding(inputs['rt'], inputs['re'], inputs['rm'])
            inp = embedded.permute(1, 0, 2)
            if self.level == 'plain':
                outputs, (hn, cn) = self.bilstm(inp, hidden)
            else:
                # Local.
                seq_len, batch_size, embed_dim = inp.size()
                outputs = Variable(torch.zeros(seq_len, batch_size, embed_dim))
                outputs = outputs.cuda() if use_cuda else outputs
                for ei in range(seq_len):
                    if ei > 0 and ei % 32 == 0:
                        # Local needs to reinit by block.
                        hidden = self.initHidden(batch_size)
                    seq_i = inp[ei, :, :].unsqueeze(0)
                    # inputs of size: (1, batch, emb_dim)
                    output, (hn, cn) = self.bilstm(seq_i, hidden)
                    outputs[ei, :, :] = output[0, :, :]
                    # output of size: (1, batch, emb_dim)
            return outputs, hn.view(self.n_layers, -1, self.hidden_size)

    def initHidden(self, batch_size):
        forward = Variable(torch.zeros(2 * self.n_layers, batch_size,
                                       self.hidden_size // 2), requires_grad=False)
        backward = Variable(torch.zeros(2 * self.n_layers, batch_size,
                                        self.hidden_size // 2), requires_grad=False)
        if use_cuda:
            return (forward.cuda(), backward.cuda())
        else:
            return (forward, backward)


class HierarchicalBiLSTMMaxPool(nn.Module):
    """"""
    def __init__(self, hidden_size, local_embed, n_layers=LAYER_DEPTH):
        super(HierarchicalBiLSTMMaxPool, self).__init__()
        self.LocalEncoder = EncoderBiLSTMMaxPool(hidden_size, local_embed,
                                                 n_layers=n_layers, level='local')
        self.GlobalEncoder = EncoderBiLSTMMaxPool(hidden_size, None,
                                                  n_layers=n_layers, level='global')


class EncoderBiLSTMMaxPool(nn.Module):
    """Vanilla encoder using pure LSTM."""
    def __init__(self, hidden_size, embedding_layer, n_layers=LAYER_DEPTH, level='plain'):
        super(EncoderBiLSTMMaxPool, self).__init__()
        self.name = 'BiLSTMMaxPool'
        self.level = level
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if self.level == 'plain' or self.level == 'local':
            self.embedding = embedding_layer
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=n_layers, bidirectional=True)

    def forward(self, inputs, hidden):
        # embedded is of size (n_batch, seq_len, emb_dim)
        # lstm needs: (seq_len, batch, input_size)
        if self.level == 'global':
            inp = inputs['local_hidden_states']
            bilstm_outs, hidden = self.bilstm(inp, hidden)
        else:
            # Local or Plain.
            embedded = self.embedding(inputs['rt'], inputs['re'], inputs['rm'])
            inp = embedded.permute(1, 0, 2)
            if self.level == 'plain':
                bilstm_outs, hidden = self.bilstm(inp, hidden)
            else:
                # Local.
                seq_len, batch_size, embed_dim = inp.size()
                bilstm_outs = Variable(torch.zeros(seq_len, batch_size, embed_dim))
                bilstm_outs = bilstm_outs.cuda() if use_cuda else bilstm_outs
                for ei in range(seq_len):
                    if ei > 0 and ei % 32 == 0:
                        hidden = self.initHidden(batch_size)
                    inputs = inp[ei, :, :].unsqueeze(0)
                    # inputs of size: (1, batch, emb_dim)
                    outputs, hidden = self.bilstm(inputs, hidden)
                    # output of size: (1, batch, emb_dim)
                    bilstm_outs[ei, :, :] = outputs[0, :, :]
        # bilstm_outs: (seq_len, batch, hidden_size * num_directions)
        output = bilstm_outs.permute(1, 2, 0)
        # bilstm_outs: (batch, hidden_size * num_directions, seq_len)
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        return bilstm_outs, output.unsqueeze(0)

    def initHidden(self, batch_size):
        forward = Variable(torch.zeros(2 * self.n_layers, batch_size,
                                       self.hidden_size // 2), requires_grad=False)
        backward = Variable(torch.zeros(2 * self.n_layers, batch_size,
                                        self.hidden_size // 2), requires_grad=False)
        if use_cuda:
            return (forward.cuda(), backward.cuda())
        else:
            return (forward, backward)


class PGenLayer(nn.Module):
    def __init__(self, emb_dim, hidden_size, enc_dim):
        super(PGenLayer, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.enc_dim = enc_dim
        self.lin = nn.Linear(self.emb_dim + self.hidden_size + self.enc_dim, 1)

    def forward(self, emb, hid, enc):
        '''
        param:  emb (batch_size, emb_dim)
                hid (batch_size, hid_dim)
                enc (batch_size, enc_dim)
        '''
        input = torch.cat((emb, hid, enc), 1)
        return F.sigmoid(self.lin(input))


class AttnDecoderRNN(nn.Module):
    """This is a plain decoder with attention."""
    def __init__(self, hidden_size, output_size, n_layers=LAYER_DEPTH, 
                 dropout_p=0.1, copy=TOCOPY):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.copy = copy

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = Attn(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        if self.copy:
            self.pgen = PGenLayer(self.hidden_size, self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)

        attn_weights = self.attn(hidden[-1, :, :], encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)

        # Adjust the dimension after bmm()
        context = context.squeeze(1)

        output = torch.cat((embedded, context), dim=1)

        # To align with the library standard (seq_len, batch, input_size)
        output = output.unsqueeze(0)
        output, nh = self.gru(output, hidden)

        output = output.squeeze(0)

        if self.copy:
            pgen = self.pgen(embedded, output, context)
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1) + pgen.log()
        else:
            pgen = 0
            # Output the final distribution
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1)

        return output, nh, context, attn_weights, pgen

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size), requires_grad=False)
        if use_cuda:
            return result.cuda()
        else:
            return result


class HierarchicalDecoder(nn.Module):
    """ The class for Hierarchical decoder.

    This module is for encapsulating the Hierarchical decoder part.

    """
    def __init__(self, hidden_size, output_size,
                 n_layers=LAYER_DEPTH, copy=TOCOPY):

        super(HierarchicalDecoder, self).__init__()
        self.global_decoder = GlobalAttnDecoderRNN(hidden_size, n_layers=n_layers)
        self.local_decoder = LocalAttnDecoderRNN(hidden_size, output_size,
                                                 n_layers=n_layers, copy=copy)


class GlobalAttnDecoderRNN(nn.Module):
    """ The class for global decoding.

    This module is for sentence-level decoding, where we calculate
    next state as S_{n}^{g} = f(S_{n-1}^{g}, S_{n-1}^{l,(T)}, C_{n}^{g})
    , and C_{n}^{g}=\sum_{j=1}^{|b|}\beta_{n,j}h_{j}^{g}

    """
    def __init__(self, hidden_size, n_layers=LAYER_DEPTH, dropout_p=0.1):
        super(GlobalAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.attn = Attn(hidden_size)

        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        attn_weights = self.attn(hidden[-1, :, :], encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)
        output = torch.cat((input, context.squeeze(1)), dim=1)

        # To align with the library standard (seq_len, batch, input_size)
        output = output.unsqueeze(0)
        output, nh = self.gru(output, hidden)

        return output, nh, context, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size), requires_grad=False)
        if use_cuda:
            return result.cuda()
        else:
            return result


class LocalAttnDecoderRNN(nn.Module):
    """ The class for local decoding.

    This module is for word-level decoding, where we calculate
    next state as S_{n}^{l,(t)}=f(S_{n}^{l,(t-1)}, y_{n}^{(t-1)}, C_{n}^{l,(t)})
    , and C_{n}^{g}=\sum_{j=1}^{|b|}\beta_{n,j}h_{j}^{g}

    """
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH,
                 n_layers=LAYER_DEPTH, dropout_p=0.1, copy=TOCOPY):
        super(LocalAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.copy = copy
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = Attn(hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)

        self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        if self.copy:
            self.pgen = PGenLayer(self.hidden_size, self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, block_attn_weights, encoder_outputs, blocks):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # blocks is a list storing tth for each block
        batch_size_blk_size, seq_len, hidden_size = encoder_outputs.size()
        batch_size = batch_size_blk_size // len(blocks)

        # calculate attention scores for each block
        hid = hidden[-1, :, :]
        hid = hid.repeat(len(blocks), 1)

        attn_weights = self.attn(hid, encoder_outputs)

        block_context = torch.bmm(attn_weights, encoder_outputs)  # (batch * blk, 1, hid)
        block_context = block_context.view(batch_size, len(blocks), hidden_size)

        context = torch.bmm(block_attn_weights, block_context)

        # Adjust the dimension after bmm()
        context = context.squeeze(1)

        output = torch.cat((embedded, context), dim=1)

        # To align with the library standard (seq_len, batch, input_size)
        output = output.unsqueeze(0)
        output, nh = self.gru(output, hidden)

        # nh = Variable(torch.zeros(hidden.size()))
        # if use_cuda:
        #     nh.cuda()

        # for i in range(self.n_layers):
        #     layer_fnc = getattr(self, "gru" + str(i))
        #     output = layer_fnc(output, hidden[i, :, :])
        #     nh[i, :, :] = output

        output = output.squeeze(0)

        if self.copy:
            pgen = self.pgen(embedded, output, context)
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1) + pgen.log()
        else:
            pgen = Variable(torch.zeros(1, 1)).cuda() if use_cuda else Variable(torch.zeros(1, 1))
            output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1)

        return output, nh, context, attn_weights, pgen

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size), requires_grad=False)
        if use_cuda:
            return result.cuda()
        else:
            return result


class Attn(nn.Module):
    """ The score function for the attention mechanism.

    We define the score function as the general function from Luong et al.
    Where score(s_{i}, h_{j}) = s_{i} * W * h_{j}

    """
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        batch_size, seq_len, hidden_size = encoder_outputs.size()
        # print(encoder_outputs.size())

        # Get hidden chuncks (batch_size, seq_len, hidden_size)
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        hiddens = hidden.repeat(1, seq_len, 1)
        attn_energies = self.score(hiddens, encoder_outputs)

        # # Calculate energies for each encoder output
        # for i in range(seq_len):
        #     attn_energies[:, i] = self.score(hidden, encoder_outputs[:, i])
        # print(attn_energies.size())

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x seq_len
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # print('size of hidden: {}'.format(hidden.size()))
        # print('size of encoder_hidden: {}'.format(encoder_output.size()))
        energy = encoder_outputs

        # batch-wise calculate dot-product
        hidden = hidden.unsqueeze(2)  # (batch, seq, 1, d)
        energy = energy.unsqueeze(3)  # (batch, seq, d, 1)

        energy = torch.matmul(hidden, energy)  # (batch, seq, 1, 1)

        # print('size of energies: {}'.format(energy.size()))

        return energy.squeeze(3).squeeze(2)
