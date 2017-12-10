"""This is core training part, containing different models."""
import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from preprocessing import data_iter
from settings import file_loc, use_cuda, MAX_LENGTH
from dataprepare import loaddata, data2index
from model import EncoderRNN, AttnDecoderRNN
from util import gettime


# TODO: 2. Extend the model


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


def sentenceloss(rt, re, rm, summary, encoder, decoder,
                 encoder_optimizer, decoder_optimizer,
                 criterion, embedding_size):
    """Function for train on sentences.

    This function will calculate the gradient and NLLloss on sentences,
    , update the model, and then return the average loss.

    """
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

    teacher_forcing_ratio = 1.0 
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
            #    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


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


def train(train_set, langs, embedding_size=600, learning_rate=0.01,
          iter_time=10, batch_size=32, show_loss=1, save_model=500):
    """The training procedure."""
    # Set the record file
    f = open('baseline.result', 'wt')

    # Set the timer
    start = time.time()

    train_iter = data_iter(train_set, batch_size=batch_size)

    # Initialize the model
    emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                       langs['rm'].n_words, embedding_size)
    emb.init_weights()

    encoder = EncoderRNN(embedding_size, emb)
    decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)

    if use_cuda:
        encoder.cuda()
        decoder.cuda()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    total_loss = 0

    for iteration in range(1, iter_time + 1):

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

        if use_cuda:
            rt, re, rm, summary = rt.cuda(), re.cuda(), rm.cuda(), summary.cuda()

        # Get the average loss on the sentences
        loss = sentenceloss(rt, re, rm, summary, encoder, decoder,
                            encoder_optimizer, decoder_optimizer, criterion,
                            embedding_size)
        total_loss += loss

        # Print the information and save model
        if iteration % show_loss == 0:
            print("Time {}, iter {}, avg loss = {:.4f}".format(
                gettime(start), iteration, total_loss / show_loss),
                file=f)
            total_loss = 0
        if iteration % save_model == 0:
            torch.save(encoder.state_dict(), "encoder_{}".format(iteration))
            torch.save(decoder.state_dict(), "decoder_{}".format(iteration))
            print("Save the model at iter {}".format(iteration), file=f)
    
    f.close()
    return encoder, decoder


def predictwords(rt, re, rm, summary, encoder, decoder, lang, embedding_size):
    """This function will predict the sentecnes given boxscore.

    Encode the given box score, decode it to sentences, and then
    return the prediction and attention matrix.

    Right now, the prediction length is limited to target length.

    """
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = summary.size()[1]

    encoder_outputs = Variable(torch.zeros(batch_length, MAX_LENGTH, embedding_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    encoder_hidden = encoder.initHidden(batch_length)

    # Encoding
    for ei in range(input_length):
        encoder_hidden = encoder(rt[:, ei], re[:, ei], rm[:, ei], encoder_hidden)

        # Store memory information
        encoder_outputs[:, ei] = encoder_hidden

    decoder_hidden = encoder_hidden

    decoder_input = Variable(torch.LongTensor(batch_length).zero_())
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(target_length, MAX_LENGTH)

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_hidden, encoder_outputs)

        # Get the attention vector at each prediction
        decoder_attentions[di] = decoder_attention.data[0][0]

        # decode the word
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == 1:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        decoder_input = Variable(topi.squeeze(1))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, valid_set, lang,
             embedding_size, iter_time=10):
    """The evaluate procedure."""
    # Get evaluate data
    valid_iter = data_iter(valid_set, batch_size=1)

    for iteration in range(iter_time):

        # Get data
        data, idx_data = get_batch(next(valid_iter))
        rt, re, rm, summary = idx_data

        # For Encoding
        rt = Variable(torch.LongTensor(rt))
        re = Variable(torch.LongTensor(re))
        rm = Variable(torch.LongTensor(rm))

        # For Decoding
        summary = Variable(torch.LongTensor(summary))

        if use_cuda:
            rt, re, rm, summary = rt.cuda(), re.cuda(), rm.cuda(), summary.cuda()

        # Get decoding words and attention matrix
        decoded_words, decoder_attentions = predictwords(rt, re, rm, summary,
                                                         encoder, decoder, lang,
                                                         embedding_size)

        print(decoded_words)


def main():
    # Default parameter
    embedding_size = 600
    learning_rate = 0.01
    train_iter_time = 100000
    batch_size = 8

    # For Training
    train_data, train_lang = loaddata(file_loc, 'train')
    train_data = data2index(train_data, train_lang)
    encoder, decoder = train(train_data, train_lang,
                             embedding_size=embedding_size, learning_rate=learning_rate,
                             iter_time=train_iter_time, batch_size=batch_size)

    # For evaluation
    valid_data, _ = loaddata(file_loc, 'valid')
    valid_data = data2index(valid_data, train_lang)
    evaluate(encoder, decoder, valid_data, train_lang['summary'], embedding_size)


if __name__ == '__main__':
    main()
