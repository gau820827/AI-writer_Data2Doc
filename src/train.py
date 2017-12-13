"""This is core training part, containing different models."""
import random
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from preprocessing import data_iter
from dataprepare import loaddata, data2index
from model import EncoderLIN, EncoderRNN, AttnDecoderRNN, docEmbedding
from util import gettime, load_model

from settings import file_loc, use_cuda, MAX_LENGTH, USE_MODEL
from settings import EMBEDDING_SIZE, LR, ITER_TIME, BATCH_SIZE
from settings import MAX_SENTENCES, ENCODER_STYLE
from settings import GET_LOSS, SAVE_MODEL, OUTPUT_FILE

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


def sentenceloss(rt, re, rm, summary, encoder, decoder,
                 encoder_optimizer, decoder_optimizer,
                 criterion, embedding_size, encoder_style):
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

    loss = 0

    # Encoding
    encoder_hidden = encoder.initHidden(batch_length)
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
          iter_time=10, batch_size=32, get_loss=GET_LOSS, save_model=SAVE_MODEL,
          encoder_style=ENCODER_STYLE, use_model=USE_MODEL):
    """The training procedure."""
    # Set the timer
    start = time.time()

    train_iter = data_iter(train_set, batch_size=batch_size)

    # Initialize the model
    emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                       langs['rm'].n_words, embedding_size)
    emb.init_weights()

    if encoder_style == 'LIN':
        encoder = EncoderLIN(embedding_size, emb)
    else:
        encoder = EncoderRNN(embedding_size, emb)

    decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)

    if use_cuda:
        encoder.cuda()
        decoder.cuda()

    if use_model is not None:
        encoder = load_model(encoder, use_model[0])
        decoder = load_model(decoder, use_model[1])

    # Different choice of optimizer
    encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate, lr_decay=0, weight_decay=0)
    decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate, lr_decay=0, weight_decay=0)
    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

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
                            embedding_size, encoder_style)
        total_loss += loss

        # Print the information and save model
        if iteration % get_loss == 0:
            print("Time {}, iter {}, avg loss = {:.4f}".format(
                gettime(start), iteration, total_loss / get_loss))
            total_loss = 0

        if iteration % save_model == 0:
            torch.save(encoder.state_dict(), "{}_encoder_{}".format(OUTPUT_FILE, iteration))
            torch.save(decoder.state_dict(), "{}_decoder_{}".format(OUTPUT_FILE, iteration))
            print("Save the model at iter {}".format(iteration))

    return encoder, decoder


def predictwords(rt, re, rm, summary, encoder, decoder, lang, embedding_size):
    """The function will predict the sentecnes given boxscore.

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

        for word in decoded_words:
            print(word, end=' ')
        print('')


def showconfig():
    """Display the configuration."""
    print("EMBEDDING_SIZE = {}\nLR = {}\nITER_TIME = {}\nBATCH_SIZE = {}".format(
        EMBEDDING_SIZE, LR, ITER_TIME, BATCH_SIZE))
    print("MAX_SENTENCES = {}\nENCODER_STYLE = {}".format(MAX_SENTENCES, ENCODER_STYLE))
    print("USE_MODEL = {}\nOUTPUT_FILE = {}".format(USE_MODEL, OUTPUT_FILE))


def main():
    # Display Configuration
    showconfig()

    # Default parameter
    embedding_size = EMBEDDING_SIZE
    learning_rate = LR
    train_iter_time = ITER_TIME
    batch_size = BATCH_SIZE

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
