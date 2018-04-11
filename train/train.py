"""This is core training part, containing different models."""
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from preprocessing import data_iter
from dataprepare import loaddata, data2index
from model import EncoderLIN, EncoderRNN, EncoderBiLSTM, GlobalEncoderRNN
from model import AttnDecoderRNN, docEmbedding
from util import gettime, load_model, showAttention
from util import PriorityQueue

from settings import file_loc, use_cuda, MAX_LENGTH, USE_MODEL
from settings import EMBEDDING_SIZE, LR, ITER_TIME, BATCH_SIZE
from settings import MAX_SENTENCES, ENCODER_STYLE
from settings import GET_LOSS, SAVE_MODEL, OUTPUT_FILE

import numpy as np

SOS_TOKEN = 0
EOS_TOKEN = 1

# TODO: Extend the model to copy-based model

# delete it:
from pprint import pprint

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
        batch_data.append(d[:2]) # keep the original data/ not indexed version
        for triplets in d[2][0]:
            for idt, t in enumerate(triplets):
                idx_data[idt].append(t)

        for idb, b in enumerate(idx_data):
            batch_idx_data[idb].append(b)

        # Calculate maximum length of the summary
        max_summary_length = max(max_summary_length, len(d[2][1]))

        batch_idx_data[3].append(d[2][1])

    return batch_data, batch_idx_data


# # # # # # # # # # # # # # # # # # # # # # # # #
# Ken edit: Hierarchical Encoder
#
def sentenceloss(rt, re, rm, summary, local_encoder, decoder, loss_optimizer,
                 criterion, embedding_size, encoder_style, 
                    global_encoder, langs):
    """
    Ken added:
        1. local_encoder
        2. global_encoder
        3. langs
    """
    """Function for train on sentences.

    This function will calculate the gradient and NLLloss on sentences,
    , update the model, and then return the average loss.

    """
    # Zero the gradient
    loss_optimizer.zero_grad()

    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = summary.size()[1]
    """
    Added by Ken
    Preprocessing the blocks:
    1. save the number of blocks in each pair -> to initialize the global encoder states
    2. save the block position for each pair
    3. <EOB> = 4
    """
    BLOCK_NUMBERS = np.ones(batch_length)
    for bi in range(batch_length):
        blocks = 0
        for ei in range(len(rm[bi][:])):
            if langs['rm'].index2word[int(rm[bi][ei])] == '<EOB>':
                BLOCK_NUMBERS[bi] += 1
    MAX_BLOCK = int(np.max(BLOCK_NUMBERS))
    local_encoder_outputs = Variable(torch.zeros(batch_length, MAX_LENGTH, embedding_size))
    local_encoder_outputs = encoder_outputs.cuda() if use_cuda else local_encoder_outputs
    # the second dimension of global encoder is the number of blocks
    global_encoder_outputs = Variable(torch.zeros(batch_length, MAX_BLOCK, embedding_size))
    global_encoder_outputs = global_encoder_outputs.cuda() if use_cuda else global_encoder_outputs
    BLOCK_JUMPS = 31
    loss = 0

    # Encoding
    if ENCODER_STYLE == 'BiLSTM':
        init_hidden = encoder.initHidden(batch_length)
        encoder_hidden, encoder_hiddens = encoder(rt, re, rm, init_hidden)

        # Store memory information
        for ei in range(input_length):
            encoder_outputs[:, ei] = encoder_hiddens[:, ei]
        """
        To do:
            Hierarchical Encoder
        """
    else:
        """
        Ken added:
        Local Encoder
            Accept input embeddings sequence.
            local_out:
                (seq_len, batch, hidden_size * num_directions)
            local_encoder_hidden:
                (num_layers * num_directions, batch, hidden_size)
                tensor containing the hidden state for t = seq_len
        """
        init_local_hidden = local_encoder.initHidden(batch_length)
        local_out, local_encoder_hidden = local_encoder(rt, re, rm, init_local_hidden)
        """
        Global Encoder
            Accept {local encoder hidden state at <EOB>} as input sequence.
            1. local_out as global_input: see dimensions above
            2. global_input:
                (block_numbers, batch, hidden_size * num_directions)
            3. BLOCK_JUMPS = 31 (for each input, the <EOB> is at the index of multiple of 31)
        """
        global_input = Variable(torch.zeros(MAX_BLOCK, batch_length, embedding_size))
        global_input = global_input.cuda() if use_cuda else global_input
        for ei in range(input_length):
            if ei % BLOCK_JUMPS == 0:
                # map ei to block number
                global_input[int(ei/(BLOCK_JUMPS+1)), :, :] = local_out[ei, :, :]
        init_global_hidden = global_encoder.initHidden(batch_length)
        global_out, global_encoder_hidden = global_encoder(global_input, init_global_hidden)
        """
        Store memory information
        Unify dimension: (batch, sequence length, hidden size)
        """
        local_encoder_outputs = local_out.permute(1,0,2)
        global_encoder_outputs = global_out.permute(1,0,2)

    # decoder starts
    decoder_hidden = decoder.initHidden(batch_length)
    decoder_hidden[0,:,:] = out[-1,:] # might be zero
    decoder_input = Variable(torch.LongTensor(batch_length).zero_())
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        loss += criterion(decoder_output, summary[:, di])
        decoder_input = summary[:, di]  # Supervised

    loss.backward()

    loss_optimizer.step()

    return loss.data[0] / target_length


def addpaddings(summary):
    """A helper function to add paddings to summary.
    Args:
        summary: A list (batch_size) of indexing summarizes.
                 [tokens]
        padding index = 2
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
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # Ken edit: Hierarchical Encoder 04/01
    #   ** Do RNN first
    #   1. add: global_encoder = GlobalEncoderRNN(embedding_size, local_encoder)
    if encoder_style == 'LIN':
        encoder = EncoderLIN(embedding_size, emb)
    elif encoder_style == 'BiLSTM':
        encoder = EncoderBiLSTM(embedding_size, emb)
    else:
        local_encoder = EncoderRNN(embedding_size, emb)
        global_encoder = GlobalEncoderRNN(embedding_size)


    decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)

    if use_cuda:
        emb.cuda()
        encoder.cuda()
        decoder.cuda()

    if use_model is not None:
        encoder = load_model(encoder, use_model[0])
        decoder = load_model(decoder, use_model[1])

    # Choose optimizer
    # Ken added opitimzer
    loss_optimizer = optim.Adam(list(local_encoder.parameters()) + list(global_encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=0)
    #decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate, lr_decay=0, weight_decay=0)

    criterion = nn.NLLLoss()

    total_loss = 0

    for iteration in range(1, iter_time + 1):
        """
        Ken added: 04/05/2018
            1. outer loop: for input x iterations
            2. global encoder update timing
        """
        # Get data
        data, idx_data = get_batch(next(train_iter))
        rt, re, rm, summary = idx_data


        # Add paddings
        rt = addpaddings(rt)
        re = addpaddings(re)
        rm = addpaddings(rm)
        summary = addpaddings(summary)
        
        rt = Variable(torch.LongTensor(rt))
        re = Variable(torch.LongTensor(re))
        rm = Variable(torch.LongTensor(rm))

        # For Decoding
        summary = Variable(torch.LongTensor(summary))

        if use_cuda:
            rt, re, rm, summary = rt.cuda(), re.cuda(), rm.cuda(), summary.cuda()

        # Get the average loss on the sentences
        # # # # # # # # # # # # # # # # # # # # # # # # #
        # calculate loss of "a batch of input sequence"
        loss = sentenceloss(rt, re, rm, summary, local_encoder, decoder,
                            loss_optimizer, criterion,
                            embedding_size, encoder_style, global_encoder, langs)
        # # # # # # # # # # # # # # # # # # # # # # # # #
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


def predictwords(rt, re, rm, summary, encoder, decoder, lang, embedding_size,
                 encoder_style, beam_size):
    """The function will predict the sentecnes given boxscore.

    Encode the given box score, decode it to sentences, and then
    return the prediction and attention matrix.

    While decoding, beam search will be conducted with default beam_size as 1.

    """
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = 1000

    encoder_outputs = Variable(torch.zeros(batch_length, MAX_LENGTH, embedding_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    encoder_hidden = encoder.initHidden(batch_length)

    # Encoding
    if encoder_style == 'BiLSTM':
        init_hidden = encoder.initHidden(batch_length)
        encoder_hidden, encoder_hiddens = encoder(rt, re, rm, init_hidden)

        # Store memory information
        for ei in range(input_length):
            encoder_outputs[:, ei] = encoder_hiddens[:, ei]

    else:
        encoder_hidden = encoder.initHidden(batch_length)
        for ei in range(input_length):
            encoder_hidden = encoder(rt[:, ei], re[:, ei], rm[:, ei], encoder_hidden)

            # Store memory information
            encoder_outputs[:, ei] = encoder_hidden

    decoder_attentions = torch.zeros(target_length, MAX_LENGTH)

    # Initialize the Beam
    # Each Beam cell contains [prob, route, decoder_hidden, atten]
    beams = [[0, [SOS_TOKEN], encoder_hidden, decoder_attentions]]

    # For each step
    for di in range(target_length):

        # For each information in the beam
        q = PriorityQueue()
        for beam in beams:

            prob, route, decoder_hidden, atten = beam
            destination = len(route) - 1

            # Get the lastest predecition
            decoder_input = route[-1]

            # If <EOS>, do not search for it
            if decoder_input == EOS_TOKEN:
                q.push(beam, prob)
                continue

            decoder_input = Variable(torch.LongTensor([decoder_input]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            # Get the attention vector at each prediction
            atten[destination] = decoder_attention.data[0][0]

            # decode the word
            topv, topi = decoder_output.data.topk(beam_size)

            for i in range(beam_size):
                p = topv[0][i]
                idp = topi[0][i]
                new_beam = [prob + p, route + [idp], decoder_hidden, atten]
                q.push(new_beam, new_beam[0])

        # Keep the highest K probability
        beams = [q.pop() for i in range(beam_size)]

        # If the highest one is finished, we take that.
        if beams[0][1][-1] == 1:
            break

    # Get decoded_words and decoder_attetntions
    decoded_words = [lang.index2word[w] for w in beams[0][1][1:]]
    decoder_attentions = beams[0][3]
    return decoded_words, decoder_attentions[:len(decoded_words)]


def evaluate(encoder, decoder, valid_set, lang,
             embedding_size, encoder_style=ENCODER_STYLE, iter_time=10,
             beam_size=1, verbose=True):
    """The evaluate procedure."""
    # Get evaluate data
    valid_iter = data_iter(valid_set, batch_size=1, shuffle=True)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()

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
                                                         embedding_size, encoder_style,
                                                         beam_size)

        res = ' '.join(decoded_words[:-1])
        if verbose:
            print(res)
        yield res

        # # FOR WRITING REPORTS ONLY
        # # Compare to the origin data
        # triplets, gold_summary = data[0]

        # for word in gold_summary:
        #     print(word, end=' ')
        # print(' ')

        # showAttention(triplets, decoded_words, decoder_attentions)


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
