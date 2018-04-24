"""This is core training part, containing different models."""
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from preprocessing import data_iter
from dataprepare import loaddata, data2index
from model import docEmbedding, Seq2Seq
from model import EncoderLIN, HierarchicalEncoderRNN, EncoderBiLSTM
from model import AttnDecoderRNN, HierarchicalDecoder
from util import gettime, load_model, showAttention
from util import PriorityQueue

from settings import file_loc, use_cuda, MAX_LENGTH, USE_MODEL
from settings import EMBEDDING_SIZE, LR, ITER_TIME, BATCH_SIZE, GRAD_CLIP
from settings import MAX_SENTENCES, ENCODER_STYLE, DECODER_STYLE
from settings import GET_LOSS, SAVE_MODEL, OUTPUT_FILE

import numpy as np

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
BLK_TOKEN = 5

# TODO: Extend the model to copy-based model


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
    for d in batch:
        idx_data = [[], [], []]  # for each triplet
        batch_data.append([d.triplets, d.summary])  # keep the original data/ not indexed version
        for triplets in d.idx_data[0]:
            for idt, t in enumerate(triplets):
                idx_data[idt].append(t)

        for idb, b in enumerate(idx_data):
            batch_idx_data[idb].append(b)

        batch_idx_data[3].append(d.idx_data[1])

    return batch_data, batch_idx_data


def find_max_block_numbers(batch_length, langs, rm):
    blocks_lens = [[0] for i in range(batch_length)]
    BLOCK_NUMBERS = np.ones(batch_length)
    for bi in range(batch_length):
        for ei in range(len(rm[bi, :])):
            if langs['rm'].index2word[int(rm[bi, ei].data[0])] == '<EOB>':
                blocks_lens[bi].append(ei)
                BLOCK_NUMBERS[bi] += 1
    return int(np.max(BLOCK_NUMBERS)), blocks_lens


def sequenceloss(rt, re, rm, summary, model):
    """Function for train on sentences.

    This function will calculate the gradient and NLLloss on sentences,
    and then return the loss.

    """
    return model.seq_train(rt, re, rm, summary)


def Hierarchical_seq_train(rt, re, rm, summary, encoder, decoder,
                           criterion, embedding_size, langs):
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = summary.size()[1]

    # MAX_BLOCK is the number of global hidden states
    # block_lens is the start position of each block
    MAX_BLOCK, blocks_lens = find_max_block_numbers(batch_length, langs, rm)
    BLOCK_JUMPS = 31

    LocalEncoder = encoder.LocalEncoder
    GlobalEncoder = encoder.GlobalEncoder

    # For now, these are redundant
    local_encoder_outputs = Variable(torch.zeros(batch_length, MAX_LENGTH, embedding_size))
    local_encoder_outputs = local_encoder_outputs.cuda() if use_cuda else local_encoder_outputs
    global_encoder_outputs = Variable(torch.zeros(batch_length, MAX_BLOCK, embedding_size))
    global_encoder_outputs = global_encoder_outputs.cuda() if use_cuda else global_encoder_outputs

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
        All inputs are compacted into dictionary
        """
        # Local Encoder set up
        init_local_hidden = LocalEncoder.initHidden(batch_length)
        local_out, local_hidden = LocalEncoder({"rt": rt, "re": re, "rm": rm},
                                               init_local_hidden)
        # Global Encoder setup
        global_input = Variable(torch.zeros(MAX_BLOCK, batch_length,
                                            embedding_size))
        global_input = global_input.cuda() if use_cuda else global_input
        for ei in range(input_length):
            if ei % BLOCK_JUMPS == 0:
                # map ei to block number
                global_input[int(ei / (BLOCK_JUMPS + 1)), :, :] = local_out[ei, :, :]

        init_global_hidden = GlobalEncoder.initHidden(batch_length)
        global_out, global_hidden = GlobalEncoder({"local_hidden_states":
                                                  global_input}, init_global_hidden)
        """
        Store memory information
        Unify dimension: (batch, sequence length, hidden size)
        """
        local_encoder_outputs = local_out.permute(1, 0, 2)
        global_encoder_outputs = global_out.permute(1, 0, 2)

    # The decoder init for developing
    global_decoder = decoder.global_decoder
    local_decoder = decoder.local_decoder

    # Currently, we pad all box-scores to be the same length and blocks
    blocks_len = blocks_lens[0]

    # decoder starts
    gnh = global_decoder.initHidden(batch_length)
    lnh = local_decoder.initHidden(batch_length)

    g_input = global_encoder_outputs[:, -1]
    l_input = Variable(torch.LongTensor(batch_length).zero_(), requires_grad=False)
    l_input = l_input.cuda() if use_cuda else l_input

    # Debugging check the dimension
    # print('hl size: {}'.format(local_encoder_outputs.size()))
    # print('gl size: {}'.format(global_encoder_outputs.size()))
    # print('global out size: {}'.format(global_out.size()))
    # print('')
    # print('g_input size: {}'.format(g_input.size()))
    # print('l_input size: {}'.format(l_input.size()))
    # print('')

    for di in range(target_length):
        # Feed the global decoder
        if di == 0 or summary[0, di].data[0] == BLK_TOKEN:
            g_output, gnh, g_context, g_attn_weights = global_decoder(
                g_input, gnh, global_encoder_outputs)

        # Feed the target as the next input
        l_output, lnh, l_context, l_attn_weights, pgen = local_decoder(
            l_input, lnh, g_attn_weights, local_encoder_outputs, blocks_len)

        idx = 0
        if local_decoder.copy:
            l_output = l_output.exp()
            prob = Variable(torch.zeros(l_output.shape), requires_grad=False)
            prob = prob.cuda() if use_cuda else prob
            if use_cuda:
                prob = prob.cuda()
            for l_attn in l_attn_weights:
                for i in range(l_attn.shape[2]):
                    prob[:, rm[:, idx + i]] += (1 - pgen) * l_attn[:, 0, i]
                idx += l_attn.shape[2]
            l_output_new = l_output + prob

            l_output_new = l_output_new.log()
        else:
            l_output_new = l_output

        loss += criterion(l_output_new, summary[:, di])
        g_input = lnh[-1, :, :]
        l_input = summary[:, di]  # Supervised

    return loss


def Plain_seq_train(rt, re, rm, summary, encoder, decoder,
                    criterion, embedding_size, langs):
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = summary.size()[1]

    encoder_outputs = Variable(torch.zeros(batch_length, MAX_LENGTH, embedding_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    # Encoding
    if ENCODER_STYLE == 'BiLSTM':
        init_hidden = encoder.initHidden(batch_length)
        encoder_hidden, encoder_hiddens = encoder(rt, re, rm, init_hidden)

        # Store memory information
        for ei in range(input_length):
            encoder_outputs[:, ei] = encoder_hiddens[:, ei]

    else:
        encoder_hidden = encoder.initHidden(batch_length)
        out, encoder_hidden = encoder(rt, re, rm, encoder_hidden)

        # Store memory information
        encoder_outputs = out.permute(1, 0, 2)

        # Get the final hidden state
        encoder_hidden = out[-1, :]

    decoder_hidden = decoder.initHidden(batch_length)
    decoder_hidden[0, :, :] = encoder_hidden  # might be zero
    decoder_input = Variable(torch.LongTensor(batch_length).zero_())
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Feed the target as the next input
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_context, decoder_attention, pgen = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
            
        if decoder.copy:
            decoder_output = decoder_output.exp()
            prob = Variable(torch.zeros(decoder_output.shape), requires_grad=False)
            prob = prob.cuda() if use_cuda else prob
            for i in range(decoder_attention.shape[2]):
                prob[:,rm[:,i]] += (1-pgen)*decoder_attention[:,0,i]

            decoder_output_new = decoder_output + prob
            decoder_output_new = decoder_output_new.log()
        else:
            decoder_output_new = decoder_output
        loss += criterion(decoder_output_new, summary[:, di])
        decoder_input = summary[:, di]  # Supervised

    return loss


def add_sentence_paddings(summarizes):
    """A helper function to add paddings to sentences.
    Args:
        summary: A list (batch_size) of indexing summarizes.
                 [tokens]
        padding index = 2
    Returns:
        A list (batch_size) with padding summarizes.
    """
    # Add block paddings
    def len_block(summary):
        return summary.count(BLK_TOKEN)

    max_blocks_length = max(list(map(len_block, summarizes)))

    for i in range(len(summarizes)):
        summarizes[i] += [BLK_TOKEN for j in range(max_blocks_length - len_block(summarizes[i]))]

    # Aligns with blocks
    def to_matrix(summary):
        mat = [[] for i in range(len_block(summary) + 1)]
        idt = 0
        for word in summary:
            mat[idt].append(word)
            if word == BLK_TOKEN:
                idt += 1
        return mat

    for i in range(len(summarizes)):
        summarizes[i] = to_matrix(summarizes[i])

    # Add sentence paddings
    def len_sentence(matrix):
        return max(list(map(len, matrix)))

    max_sentence_length = max([len_sentence(s) for s in summarizes])
    for i in range(len(summarizes)):
        for j in range(len(summarizes[i])):
            summarizes[i][j] += [PAD_TOKEN for k in range(max_sentence_length - len(summarizes[i][j]))]

    # Join back the matrix
    def to_list(matrix):
        return [j for i in matrix for j in i]

    for i in range(len(summarizes)):
        summarizes[i] = to_list(summarizes[i])

    return summarizes


def addpaddings(tokens):
    """A helper function to add paddings to tokens.

    Args:
        summary: A list (batch_size) of indexing tokens.

    Returns:
        A list (batch_size) with padding tokens.
    """
    max_length = len(max(tokens, key=len))
    for i in range(len(tokens)):
        tokens[i] += [PAD_TOKEN for i in range(max_length - len(tokens[i]))]
    return tokens


def train(train_set, langs, embedding_size=600, learning_rate=0.01,
          iter_time=10, batch_size=32, get_loss=GET_LOSS, save_model=SAVE_MODEL,
          encoder_style=ENCODER_STYLE, decoder_style=DECODER_STYLE,
          use_model=USE_MODEL):
    """The training procedure."""
    # Set the timer
    start = time.time()

    # Initialize the model
    emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                       langs['rm'].n_words, embedding_size)
    emb.init_weights()

    # Choose encoder style
    # TODO:: Set up a choice for hierarchical or not
    if encoder_style == 'LIN':
        encoder = EncoderLIN(embedding_size, emb)
    elif encoder_style == 'BiLSTM':
        encoder = EncoderBiLSTM(embedding_size, emb)
    else:
        # initialize hierarchical encoder rnn, (both global and local)
        encoder_args = {"hidden_size": embedding_size, "local_embed": emb}
        encoder = HierarchicalEncoderRNN(**encoder_args)

    # Choose decoder style and training function
    if decoder_style == 'HierarchicalRNN':
        decoder = HierarchicalDecoder(embedding_size, langs['summary'].n_words)
        train_func = Hierarchical_seq_train
    else:
        decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words)
        train_func = Plain_seq_train

    if use_cuda:
        emb.cuda()
        encoder.cuda()
        decoder.cuda()
    
    # Choose optimizer
    #loss_optimizer = optim.Adagrad(list(encoder.parameters()) + list(decoder.parameters()),
    #                               lr=learning_rate, lr_decay=0, weight_decay=0)
    loss_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)


    # loss_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    if use_model is not None:
        encoder = load_model(encoder, use_model[0])
        decoder = load_model(decoder, use_model[1])
        loss_optimizer.load_state_dict(torch.load(use_model[2]))

    criterion = nn.NLLLoss()

    # Build up the model
    model = Seq2Seq(encoder, decoder, train_func, criterion, embedding_size, langs)

    print(encoder)
    print(decoder)
    print(loss_optimizer)

    total_loss = 0
    iteration = 0
    for epo in range(1, iter_time + 1):
        # Start of an epoch
        print("Epoch #%d" % (epo))

        # Get data
        train_iter = data_iter(train_set, batch_size=batch_size)
        for dt in train_iter:
            iteration += 1
            data, idx_data = get_batch(dt)
            rt, re, rm, summary = idx_data

            # Add paddings
            rt = addpaddings(rt)
            re = addpaddings(re)
            rm = addpaddings(rm)

            # For summary paddings, if the model is herarchical then pad between sentences
            if decoder_style == 'HierarchicalRNN':
                summary = add_sentence_paddings(summary)
            else:
                summary = addpaddings(summary)

            rt = Variable(torch.LongTensor(rt), requires_grad=False)
            re = Variable(torch.LongTensor(re), requires_grad=False)
            rm = Variable(torch.LongTensor(rm), requires_grad=False)

            # For Decoding
            summary = Variable(torch.LongTensor(summary), requires_grad=False)

            if use_cuda:
                rt, re, rm, summary = rt.cuda(), re.cuda(), rm.cuda(), summary.cuda()

            # Zero the gradient
            loss_optimizer.zero_grad()

            # calculate loss of "a batch of input sequence"
            loss = sequenceloss(rt, re, rm, summary, model)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm(list(model.encoder.parameters()) + list(model.decoder.parameters()), GRAD_CLIP)
            loss_optimizer.step()

            # Get the average loss on the sentences
            target_length = summary.size()[1]
            total_loss += loss.data[0] / target_length

            # Print the information and save model
            if iteration % get_loss == 0:
                print("Time {}, iter {}, avg loss = {:.4f}".format(
                    gettime(start), iteration, total_loss / get_loss))
                total_loss = 0

        if epo % save_model == 0:
            torch.save(encoder.state_dict(),
                    "{}_encoder_{}".format(OUTPUT_FILE, iteration))
            torch.save(decoder.state_dict(),
                    "{}_decoder_{}".format(OUTPUT_FILE, iteration))
            torch.save(loss_optimizer.state_dict(),
                    "models/{}_optim_{}".format(OUTPUT_FILE, iteration))
            print("Save the model at iter {}".format(iteration))

    return model.encoder, model.decoder


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
        out, encoder_hidden = encoder(rt, re, rm, encoder_hidden)

        # Store memory information
        encoder_outputs = out.permute(1, 0, 2)

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
            atten[destination, :decoder_attention.shape[2]] = decoder_attention.data[0, 0, :]

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
    print("MAX_SENTENCES = {}\nGRAD_CLIP = {}".format(MAX_SENTENCES, GRAD_CLIP))
    print("DECODER_STYLE = {}\nENCODER_STYLE = {}".format(DECODER_STYLE, ENCODER_STYLE))
    print("USE_MODEL = {}\nOUTPUT_FILE = {}".format(USE_MODEL, OUTPUT_FILE))


def main():
    print("Start Training")
    # Display Configuration
    showconfig()
    # Default parameter
    embedding_size = EMBEDDING_SIZE
    learning_rate = LR
    train_iter_time = ITER_TIME
    batch_size = BATCH_SIZE

    # For Training
    train_data, train_lang = loaddata(file_loc, 'train', 200)
    train_data = data2index(train_data, train_lang)
    encoder, decoder = train(train_data, train_lang,
                             embedding_size=embedding_size, learning_rate=learning_rate,
                             iter_time=train_iter_time, batch_size=batch_size, use_model=USE_MODEL)

    # For evaluation
    valid_data, _ = loaddata(file_loc, 'valid')
    valid_data = data2index(valid_data, train_lang)
    evaluate(encoder, decoder, valid_data, train_lang['summary'], embedding_size)


if __name__ == '__main__':
    main()
