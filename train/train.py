"""This is core training part, containing different models."""
import time
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from preprocessing import data_iter
from dataprepare import loaddata, data2index
from model import docEmbedding, Seq2Seq
from model import EncoderLIN, EncoderBiLSTM, EncoderBiLSTMMaxPool, EncoderRNN
from model import HierarchicalLIN, HierarchicalRNN
from model import HierarchicalBiLSTM, HierarchicalBiLSTMMaxPool
from model import AttnDecoderRNN, HierarchicalDecoder
from util import gettime, load_model, show_triplets

from settings import file_loc, use_cuda
from settings import EMBEDDING_SIZE, LR, EPOCH_TIME, BATCH_SIZE, GRAD_CLIP
from settings import MAX_SENTENCES, ENCODER_STYLE, DECODER_STYLE, TOCOPY
from settings import GET_LOSS, SAVE_MODEL, OUTPUT_FILE, COPY_PLAYER, MAX_LENGTH
from settings import LAYER_DEPTH, PRETRAIN, MAX_TRAIN_NUM, iterNum

import numpy as np

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
EOB_TOKEN = 4
BLK_TOKEN = 5


def get_batch(batch):
    """Get the batch into training format.

    Arg:
        batch: The iterator of the dataset

    Returns:
        batch_data: The origin processed data
                    (i.e batch_size * (triples[rt, re, rm, orm], summary, osummary))
        batch_idx_data: The indexing-processed data
                        (e.g batch_size * (r.t, r.e, r.m, summary, osummary))

    """
    batch_data = []
    batch_idx_data = [[], [], [], [], [], []]
    for d in batch:
        idx_data = [[], [], [], []]  # for each triplet
        batch_data.append([d.triplets, d.summary])  # keep the original data/ not indexed version
        for triplets in d.idx_data[0]:
            for idt, t in enumerate(triplets):
                idx_data[idt].append(t)

        for idb, b in enumerate(idx_data):
            batch_idx_data[idb].append(b)
        batch_idx_data[4].append(d.idx_data[1])
        batch_idx_data[5].append(d.idx_data[2])

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


def initGlobalEncoderInput(MAX_BLOCK, batch_length, input_length, embedding_size,
                           local_outputs, BLOCK_JUMPS=32, name=None):
    """
    Args: local_outputs: (batch, seq_len, embed_size)
    """
    # print("Max block = ", MAX_BLOCK)
    # print("input length = ", input_length)
    if name == 'LIN':
        return local_outputs
    global_input = Variable(torch.zeros(MAX_BLOCK, batch_length,
                                        embedding_size))
    global_input = global_input.cuda() if use_cuda else global_input
    for ei in range(1, input_length + 1):
        # In this way, the first global state is the 32 of local state
        if ei % BLOCK_JUMPS == 0:
            block_idx = int(ei / (BLOCK_JUMPS + 1))
            global_input[block_idx, :, :] = local_outputs[ei - 1, :, :]
            # print("ei = {}, local {} put in block number = {}"
            #       .format(ei, ei - 1, block_idx))
    return global_input


def sequenceloss(rt, re, rm, orm, summary, data, model):
    """Function for train on sentences.

    This function will calculate the gradient and NLLloss on sentences,
    and then return the loss.

    """
    return model.seq_train(rt, re, rm, orm, summary, data)


def Hierarchical_seq_train(rt, re, rm, orm, summary, data, encoder, decoder,
                           criterion, embedding_size, langs, oov_dict):
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = summary.size()[1]
    # MAX_BLOCK is the number of global hidden states
    # block_lens is the start position of each block
    MAX_BLOCK, blocks_lens = find_max_block_numbers(batch_length, langs, rm)

    inputs = {"rt": rt, "re": re, "rm": rm}

    LocalEncoder = encoder.LocalEncoder
    GlobalEncoder = encoder.GlobalEncoder

    loss = 0

    # Encoding: Encoder output all has (seq_len, batch, hid_size)
    init_local_hidden = LocalEncoder.initHidden(batch_length)
    init_global_hidden = GlobalEncoder.initHidden(batch_length)
    local_encoder_outputs, local_hidden = LocalEncoder(inputs, init_local_hidden)
    global_input = initGlobalEncoderInput(MAX_BLOCK, batch_length, input_length,
                                          embedding_size, local_encoder_outputs,
                                          name=GlobalEncoder.name)
    global_encoder_outputs, global_hidden = GlobalEncoder({"local_hidden_states":
                                                          global_input}, init_global_hidden)

    """
    Encoder Final Dimension: (batch, sequence length, hidden size)
    """
    local_encoder_outputs = local_encoder_outputs.permute(1, 0, 2)
    global_encoder_outputs = global_encoder_outputs.permute(1, 0, 2)

    # Debugging: Test encoder outputs
    # print("Local Encoder shape: ", local_encoder_outputs.shape)
    # print("Global Encoder shape: ", global_encoder_outputs.shape)

    # The decoder init for developing
    global_decoder = decoder.global_decoder
    local_decoder = decoder.local_decoder

    # Currently, we pad all box-scores to be the same length and blocks
    blocks_len = blocks_lens[0]

    """
    g_input_{0} should be 0 vector with dim (batch, hidden)
    gnh should be the last hidden state of global encoder
    """
    g_input = global_decoder.initHidden(batch_length).permute(1, 2, 0)[:, :, -1]
    gnh = global_hidden

    # l_input_{0} should also be 0 vector with dim (batch) -> 0 as <SOS>
    l_input = Variable(torch.LongTensor(batch_length).zero_(), requires_grad=False)
    l_input = l_input.cuda() if use_cuda else l_input

    # This is redundant, we will replace this after time stamp 0 anyway
    lnh = local_decoder.initHidden(batch_length)

    # Debugging: check the dimension
    # print('hl size: {}'.format(local_encoder_outputs.size()))
    # print('gl size: {}'.format(global_encoder_outputs.size()))
    # print('global out size: {}'.format(global_out.size()))
    # print('')
    # print('g_input size: {}'.format(g_input.size()))
    # print('l_input size: {}'.format(l_input.size()))
    # print('')

    # Reshape the local_encoder outputs to (batch * blocks, blk_size, hidden)
    local_encoder_outputs = local_encoder_outputs.contiguous().view(batch_length * len(blocks_len),
                                                                    input_length // len(blocks_len),
                                                                    embedding_size)
    for di in range(target_length):
        # Feed the global decoder
        if di == 0 or l_input[0].data[0] == BLK_TOKEN:
            g_output, gnh, g_context, g_attn_weights = global_decoder(
                g_input, gnh, global_encoder_outputs)

            # Reset the local init status
            lnh = gnh

        # Feed the target as the next input
        l_output, lnh, l_context, l_attn_weights, pgen = local_decoder(
            l_input, lnh, g_attn_weights, local_encoder_outputs, blocks_len)

        if local_decoder.copy:
            # print(l_attn_weights.size())  # [batch * blocks, 1, blk_size]
            # print(g_attn_weights.size())  # [batch, 1, blocks]
            l_attn_weights = l_attn_weights.squeeze(1)
            bg_attn_weights = g_attn_weights.view(batch_length * len(blocks_len), -1)

            # batch-wise calculation for block attentions
            # (batch * blocks, blk_size) * (batch * blocks, 1)
            combine_attn_weights = l_attn_weights * bg_attn_weights

            combine_attn_weights = combine_attn_weights.view(batch_length, -1)

            # print(l_output)  # [batch, vocb_lang]
            prob = Variable(torch.zeros(l_output.shape))
            prob = prob.cuda() if use_cuda else prob

            # Now we had rm as (batch, input) and combine_attn_weights as (batch, input)
            # Add up to the pgen probability matrix
            prob = prob.scatter_add(1, orm, combine_attn_weights)

            # Check <UNK> equality
            curword = summary[0, di].data[0]
            oov_exist = False
            if curword == 3:
                prob_oov = Variable(torch.zeros([1,1]))
                prob_oov += 1e-10 # TODO: find a way to fix zero prob
                prob_oov = prob_oov.cuda() if use_cuda else prob_oov
                for i in range(combine_attn_weights.shape[1]):
                    if data[0][0][i][2] == data[0][1][di]:
                        oov_exist = True
                        prob_oov += combine_attn_weights[0,i]
                prob_oov *= (1-pgen)
            # if oov_exist:
            #     print(prob_oov)

            l_output_new = (l_output.exp() + (1 - pgen) * prob).log()

            if oov_exist:
                idx = Variable(torch.LongTensor([0]))
                idx = idx.cuda() if use_cuda else idx
                loss += criterion(prob_oov.log(), idx)
            else:
                loss += criterion(l_output_new, summary[:, di])
        else:
            l_output_new = l_output
            loss += criterion(l_output_new, summary[:, di])

        g_input = lnh[-1, :, :]
        l_input = summary[:, di]  # Supervised

    return loss


def Plain_seq_train(rt, re, rm, orm, summary, data, encoder, decoder,
                    criterion, embedding_size, langs, oov_dict):
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = summary.size()[1]

    encoder_outputs = Variable(torch.zeros(batch_length, input_length, embedding_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    # Encoding
    init_hidden = encoder.initHidden(batch_length)
    inputs = {"rt": rt, "re": re, "rm": rm}
    encoder_outputs, encoder_hiddens = encoder(inputs, init_hidden)

    # encoder_outputs: (seq_len, batch_size, hidden_dim)

    context_vec = encoder_outputs[-1, :, :]
    # context_vec: (batch_size, hidden_dim)
    encoder_outputs = encoder_outputs.permute(1, 0, 2)

    decoder_hidden = decoder.initHidden(batch_length)
    decoder_hidden[0, :, :] = context_vec  # might be zero
    decoder_input = Variable(torch.LongTensor(batch_length).zero_())
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Feed the target as the next input
    copy_ctr = 0
    for di in range(target_length):

        decoder_output, decoder_hidden, decoder_context, decoder_attention, pgen = decoder(

            decoder_input, decoder_hidden, encoder_outputs)

        if decoder.copy:
            prob = Variable(torch.zeros(decoder_output.shape))
            prob = prob.cuda() if use_cuda else prob

            decoder_attention = decoder_attention.squeeze(1)
            prob = prob.scatter_add(1, orm, decoder_attention)
            # reset <UNK> prob
            # calculate oov prob.
            # prob[:,3] = 0

            # Check <UNK> equality
            curword = summary[0, di].data[0]
            oov_exist = False
            if curword == 3:
                prob_oov = Variable(torch.zeros([1,1]))
                prob_oov += 1e-10 # TODO: find a way to fix zero prob
                prob_oov = prob_oov.cuda() if use_cuda else prob_oov
                for i in range(decoder_attention.shape[1]):
                    if data[0][0][i][2] == data[0][1][di]:
                        oov_exist = True
                        prob_oov += decoder_attention[0,i]
                prob_oov *= (1-pgen)
            
            decoder_output_new = (decoder_output.exp() + (1-pgen)*prob).log()
            # print(torch.sum(decoder_output_new.exp(), 1))
            #loss += criterion((1-pgen).log(), orm)
            if oov_exist:
                copy_ctr+=1
                idx = Variable(torch.LongTensor([0]))
                idx = idx.cuda() if use_cuda else idx
                loss += criterion(prob_oov.log(), idx)
            else:
                loss += criterion(decoder_output_new, summary[:, di])            
        else:
            decoder_output_new = decoder_output
            loss += criterion(decoder_output_new, summary[:, di])
        
        decoder_input = summary[:, di]  # Supervised
    print("Copy_training: {}".format(copy_ctr))
    return loss


def add_sentence_paddings(summarizes, osummarizes):
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
        osummarizes[i] += [6 for j in range(max_blocks_length - len_block(summarizes[i]))]        
        summarizes[i] += [BLK_TOKEN for j in range(max_blocks_length - len_block(summarizes[i]))]

    # Aligns with blocks, and remove <BLK> at this time
    def to_matrix(summary, osummary):
        mat = [[] for i in range(len_block(summary) + 1)]
        omat = [[] for i in range(len_block(summary) + 1)]
        idt = 0
        for i,word in enumerate(summary):
            if word == BLK_TOKEN:
                idt += 1
            else:
                mat[idt].append(word)
                omat[idt].append(osummary[i])
        return mat, omat 

    for i in range(len(summarizes)):
        summarizes[i], osummarizes[i] = to_matrix(summarizes[i], osummarizes[i])

    # Add sentence paddings
    def len_sentence(matrix):
        return max(list(map(len, matrix)))

    max_sentence_length = max([len_sentence(s) for s in summarizes])
    for i in range(len(summarizes)):
        for j in range(len(summarizes[i])):
            osummarizes[i][j] += [6 for k in range(max_sentence_length - len(summarizes[i][j]))]
            osummarizes[i][j] += [6]
            
            summarizes[i][j] += [PAD_TOKEN for k in range(max_sentence_length - len(summarizes[i][j]))]
            summarizes[i][j] += [BLK_TOKEN]

    # Join back the matrix
    def to_list(matrix):
        return [j for i in matrix for j in i]

    for i in range(len(summarizes)):
        summarizes[i] = to_list(summarizes[i])
        osummarizes[i] = to_list(osummarizes[i])

    return summarizes, osummarizes


def addpaddings(tokens, company=None, to=None):
    """A helper function to add paddings to tokens.

    Args:
        summary: A list (batch_size) of indexing tokens.

    Returns:
        A list (batch_size) with padding tokens.
    """
    max_length = len(max(tokens, key=len))
    for i in range(len(tokens)):
        if to is not None:
            tokens[i] += [to for i in range(max_length - len(tokens[i]))]
        else:
            if company is not None:
                company[i][1] += [PAD_TOKEN for i in range(max_length - len(tokens[i]))]
                
            tokens[i] += [PAD_TOKEN for i in range(max_length - len(tokens[i]))]
    if company is not None:
        return tokens, company
    else:
        return tokens

def model_initialization(encoder_style, 
decoder_style, langs, 
embedding_size, learning_rate, pretrain, layer_depth, to_copy, iter_num, load_optim=True):
    # Initialize the model
    emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                       langs['rm'].n_words, embedding_size)
    emb.init_weights()

    # Choose encoder style
    if encoder_style == 'LIN':
        encoder = EncoderLIN(embedding_size, emb)
    
    elif encoder_style == 'RNN':
        encoder = EncoderRNN(embedding_size, emb)

    elif encoder_style == 'RNN':
        encoder = EncoderRNN(embedding_size, emb)

    elif encoder_style == 'BiLSTM':
        encoder = EncoderBiLSTM(embedding_size, emb, n_layers=layer_depth)

    elif encoder_style == 'BiLSTMMaxPool':
        encoder = EncoderBiLSTMMaxPool(embedding_size, emb, n_layers=layer_depth)

    elif encoder_style == 'HierarchicalBiLSTM':
        encoder_args = {"hidden_size": embedding_size, "local_embed": emb,
                        "n_layers": layer_depth}
        encoder = HierarchicalBiLSTM(**encoder_args)

    elif encoder_style == 'HierarchicalBiLSTMMaxPool':
        encoder_args = {"hidden_size": embedding_size, "local_embed": emb,
                        "n_layers": layer_depth}
        encoder = HierarchicalBiLSTMMaxPool(**encoder_args)

    elif encoder_style == 'HierarchicalLIN':
        encoder_args = {"hidden_size": embedding_size, "local_embed": emb}
        encoder = HierarchicalLIN(**encoder_args)

    else:
        # initialize hierarchical encoder rnn, (both global and local)
        encoder_args = {"hidden_size": embedding_size, "local_embed": emb,
                        "n_layers": layer_depth}
        encoder = HierarchicalRNN(**encoder_args)

    # Choose decoder style and training function
    if decoder_style == 'HierarchicalRNN':
        decoder = HierarchicalDecoder(embedding_size, langs['summary'].n_words,
                                      n_layers=layer_depth, copy=to_copy)
        train_func = Hierarchical_seq_train
    else:
        decoder = AttnDecoderRNN(embedding_size, langs['summary'].n_words,
                                 n_layers=layer_depth, copy=to_copy)
        train_func = Plain_seq_train

    if use_cuda:
        emb.cuda()
        encoder.cuda()
        decoder.cuda()

    # Choose optimizer
    loss_optimizer = optim.Adagrad(list(encoder.parameters()) + list(decoder.parameters()),
                                   lr=learning_rate, lr_decay=0, weight_decay=0)

    # loss_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
    #                             lr=learning_rate)

    # Load pre-train model
    use_model = None
    if pretrain is not None and iter_num is not None:
        use_model = ['./models/' + pretrain + '_' + s + '_' + str(iter_num)
                     for s in ['encoder', 'decoder', 'optim']]

    if use_model is not None:
        encoder = load_model(encoder, use_model[0])
        decoder = load_model(decoder, use_model[1])
        if load_optim:
            if not use_cuda:
                loss_optimizer.load_state_dict(torch.load(use_model[2], map_location=lambda storage, loc: storage))
            else:
                loss_optimizer.load_state_dict(torch.load(use_model[2]))
        else:
            loss_optimizer = None
    print(use_model)
    return encoder, decoder, loss_optimizer, train_func


def train(train_set, langs, oov_dict, embedding_size=EMBEDDING_SIZE, learning_rate=LR,
          batch_size=BATCH_SIZE, get_loss=GET_LOSS, grad_clip=GRAD_CLIP,
          encoder_style=ENCODER_STYLE, decoder_style=DECODER_STYLE,
          to_copy=TOCOPY, epoch_time=EPOCH_TIME, layer_depth=LAYER_DEPTH,
          max_length=MAX_LENGTH, max_sentence=MAX_SENTENCES,
          save_model=SAVE_MODEL, output_file=OUTPUT_FILE,
          iter_num=iterNum, pretrain=PRETRAIN):
    """The training procedure."""
    # # Test arg parser (For Debugging)
    # print("embedding_size={}, learning_rate={}, batch_size={}, get_loss={}, grad_clip={},\
    #         encoder_style={}, decoder_style={}, max_length={},\
    #         max_sentece={}, save_model={}, output_file={}, to_copy={},\
    #         epoch={}, layer_depth={}, iter num={}, pretrain={}".format(
    #         embedding_size, learning_rate, batch_size, get_loss, grad_clip,
    #         encoder_style, decoder_style, max_length, max_sentece, save_model, output_file,
    #         to_copy, epoch_time, layer_depth, iter_num, pretrain))
    # Set the timer
    start = time.time()

    encoder, decoder, loss_optimizer, train_func = model_initialization(encoder_style, 
                                            decoder_style, langs, 
                                            embedding_size, learning_rate, pretrain, layer_depth=layer_depth, to_copy=to_copy, iter_num=iter_num)

    criterion = nn.NLLLoss()

    # Build up the model
    model = Seq2Seq(encoder, decoder, train_func, None, criterion, embedding_size, langs, oov_dict)

    # print(encoder)
    # print(decoder)
    # print(loss_optimizer)

    total_loss = 0
    iteration = 0
    for epo in range(1, epoch_time + 1):
        # Start of an epoch
        print("Epoch #%d" % (epo))

        # Get data
        train_iter = data_iter(train_set, batch_size=batch_size)
        for dt in train_iter:
            iteration += 1
            data, idx_data = get_batch(dt)
            rt, re, rm, orm, summary, osummary = idx_data
            # print(summary[0])
            # print(osummary[0])
            # print(len(summary[0]))
            # print(len(osummary[0]))

            # for i,idx in enumerate(summary[0]):
            #     if idx != 3:
            #         print('{}'.format(langs['summary'].index2word[idx]), end=' ')
            #     else:
            #         print('{}'.format(oov_dict.index2word[osummary[0][i]]), end=' ')
                    
            # print(data[0][1])

            # exit(1)

            # Debugging: check the input triplets
            # show_triplets(data[0][0])
            # Add paddings
            rt = addpaddings(rt)
            re = addpaddings(re)
            rm = addpaddings(rm)
            orm = addpaddings(orm)


            # For summary paddings, if the model is herarchical then pad between sentences
            # If the batch_size is 1 then we don't need to do sentence padding
            if decoder_style == 'HierarchicalRNN' and batch_size != 1:
                summary, osummary = add_sentence_paddings(summary, osummary)
            else:
                summary, data = addpaddings(summary, data)
                #osummary = addpaddings(osummary, to=6)

            rt = Variable(torch.LongTensor(rt), requires_grad=False)
            re = Variable(torch.LongTensor(re), requires_grad=False)
            rm = Variable(torch.LongTensor(rm), requires_grad=False)
            orm = Variable(torch.LongTensor(orm), requires_grad=False)
            
            # DEBUG:
            #if torch.sum(rm==3).item() == 0:
            #    continue

            # For Decoding
            summary = Variable(torch.LongTensor(summary), requires_grad=False)
            #osummary = Variable(torch.LongTensor(osummary), requires_grad=False)
            #print(summary)
            #print(osummary)

            if use_cuda:
                rt, re, rm, orm, summary = rt.cuda(), re.cuda(), rm.cuda(), orm.cuda(), summary.cuda()

            # Zero the gradient
            loss_optimizer.zero_grad()
            model.train()

            # calculate loss of "a batch of input sequence"
            loss = sequenceloss(rt, re, rm, orm, summary, data, model)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm(list(model.encoder.parameters()) +
                                          list(model.decoder.parameters()),
                                          grad_clip)
            loss_optimizer.step()

            # Get the average loss on the sentences
            target_length = summary.size()[1]
            if float(torch.__version__[:3]) > 0.3:
                total_loss += loss.item() / target_length
            else:
                total_loss += loss.data[0] / target_length

            # Print the information and save model
            if iteration % get_loss == 0:
                print("Time {}, iter {}, Seq_len:{}, avg loss = {:.4f}".format(
                    gettime(start), iteration, target_length, total_loss / get_loss))
                total_loss = 0

        if epo % save_model == 0:
            torch.save(encoder.state_dict(),
                       "models/{}_encoder_{}".format(output_file, iteration))
            torch.save(decoder.state_dict(),
                       "models/{}_decoder_{}".format(output_file, iteration))
            torch.save(loss_optimizer.state_dict(),
                       "models/{}_optim_{}".format(output_file, iteration))
            print("Save the model at iter {}".format(iteration))

    return model.encoder, model.decoder


def setupconfig(args):
    """Set up and display the configuration."""
    # TODO: restricted lin for layer = 1
    parameters = {}
    for arg in vars(args):
        parameters[arg] = getattr(args, arg)
        # print("{} = {}".format(arg, parameters[arg]))
    print("---------------")
    print("Parameter Settings:")
    hierarchical_choices = ['HierarchicalRNN', 'HierarchicalBiLSTM',
                            'HierarchicalLIN']
    plain_choices = ['LIN', 'BiLSTM', 'RNN', 'BiLSTMMax']

    if parameters['encoder_style'] in hierarchical_choices and parameters['decoder_style'] != 'HierarchicalRNN':
        print("You must give me two hierarchical NNs!!!!!!!!!")
        quit()

    if parameters['encoder_style'] in plain_choices and parameters['decoder_style'] != 'RNN':
        print("You must give me two plain NNs!!!!!!!!!")
        quit()

    if (parameters['encoder_style'] == 'LIN' or parameters['encoder_style'] == 'HierarchicalLIN') and parameters['layer_depth'] != 1:
        print("Linear encoder only has depth = 1, adjust layer to 1.")
        parameters['layer_depth'] = 1

    copy_player = COPY_PLAYER
    for arg in parameters:
        if arg == 'copy_player':
            if parameters[arg] == 'True':
                copy_player = True
        print("{} = {}".format(arg, parameters[arg]))
    print("---------------")
    parameters.pop('copy_player', None)

    return parameters, copy_player


def main(args):
    """Main train driver."""
    print("Start Training")

    parameters, copy_player = setupconfig(args)

    if parameters['batch_size'] > 1:
        print("Currently only support batch_size == 1")
        exit(1)

    # For Training
    train_data, train_lang = loaddata(file_loc, 'train',
                                      copy_player=copy_player)

    if parameters['max_train_nums'] is not None:
        mx_train = parameters['max_train_nums']
        train_data = train_data[:mx_train]
    del(parameters['max_train_nums'])

    train_data, oov_dict = data2index(train_data, train_lang, max_sentences=parameters['max_sentence'])

    encoder, decoder = train(train_data, train_lang, oov_dict, **parameters)

    # For evaluation
    # valid_data, _ = loaddata(file_loc, 'valid',
    #                         copy_player=copy_player)

    # valid_data = data2index(valid_data, train_lang)
    # evaluate(encoder, decoder, valid_data, train_lang['summary'],
    #         parameters['embedding_size'])


def parse_argument():
    """Hyperparmeter tuning."""
    encoder_choices = ['LIN', 'BiLSTM', 'RNN', 'BiLSTMMaxPool',
                       'HierarchicalRNN', 'HierarchicalBiLSTMMaxPool',
                       'HierarchicalBiLSTM', 'HierarchicalLIN']

    decoder_choices = ['RNN', 'HierarchicalRNN']

    ap = argparse.ArgumentParser()
    ap.add_argument("-embed", "--embedding_size",
                    type=int, default=EMBEDDING_SIZE)

    ap.add_argument("-lr", "--learning_rate",
                    type=float, default=LR)

    ap.add_argument("-batch", "--batch_size",
                    type=int, default=BATCH_SIZE)

    ap.add_argument("-getloss", "--get_loss", type=int,
                    default=GET_LOSS)

    ap.add_argument("-encoder", "--encoder_style",
                    choices=encoder_choices, default=ENCODER_STYLE)

    ap.add_argument("-decoder", "--decoder_style",
                    choices=decoder_choices, default=DECODER_STYLE)

    ap.add_argument("-epochsave", "--save_model", type=int, default=SAVE_MODEL)

    ap.add_argument("-outputfile", "--output_file", default=OUTPUT_FILE)

    ap.add_argument("-copy", "--to_copy", choices=['True', 'False'],
                    default=TOCOPY)

    ap.add_argument("-copyplayer", "--copy_player", choices=['True', 'False'],
                    default=COPY_PLAYER)

    ap.add_argument("-gradclip", "--grad_clip", type=int, default=GRAD_CLIP)

    ap.add_argument("-pretrain", "--pretrain", default=PRETRAIN)

    ap.add_argument("-iternum", "--iter_num", default=iterNum)

    ap.add_argument("-layer", "--layer_depth", type=int, default=LAYER_DEPTH)

    ap.add_argument("-epoch", "--epoch_time", type=int, default=EPOCH_TIME)

    ap.add_argument("-maxlength", "--max_length", type=int, default=MAX_LENGTH)

    # max_sentence is optional
    ap.add_argument("-maxsentence", "--max_sentence", type=int, default=MAX_SENTENCES)

    ap.add_argument("-maxtrain", "--max_train_nums", type=int, default=MAX_TRAIN_NUM)

    return ap.parse_args()

if __name__ == '__main__':
    args = parse_argument()
    main(args)
