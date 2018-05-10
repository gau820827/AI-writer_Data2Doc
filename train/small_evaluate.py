"""Evaluate the model."""
import time

import torch
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from preprocessing import data_iter
from dataprepare import loaddata, data2index
from train import get_batch, model_initialization, addpaddings
from train import find_max_block_numbers, initGlobalEncoderInput, add_sentence_paddings

from model import docEmbedding, Seq2Seq
from model import EncoderLIN, EncoderBiLSTM, EncoderBiLSTMMaxPool, EncoderRNN
from model import HierarchicalRNN, HierarchicalBiLSTM, HierarchicalLIN
from model import AttnDecoderRNN, HierarchicalDecoder
from util import PriorityQueue, gettime


from settings import file_loc, use_cuda
from settings import EMBEDDING_SIZE, ENCODER_STYLE, DECODER_STYLE
from settings import LAYER_DEPTH, TOCOPY, PRETRAIN, iterNum

from util import load_model

# TODO move token numbering to a common file
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
EOB_TOKEN = 4
BLK_TOKEN = 5

def hierarchical_predictwords(rt, re, rm, orm, data, encoder, decoder, embedding_size, langs, oov_dict, beam_size):
    """The function will predict the sentecnes given boxscore.

    Encode the given box score, decode it to sentences, and then
    return the prediction and attention matrix.

    While decoding, beam search will be conducted with default beam_size as 1.

    """
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = 1000

    # MAX_BLOCK is the number of global hidden states
    # block_lens is the start position of each block
    MAX_BLOCK, blocks_lens = find_max_block_numbers(batch_length, langs, rm)

    inputs = {"rt": rt, "re": re, "rm": rm}

    LocalEncoder = encoder.LocalEncoder
    GlobalEncoder = encoder.GlobalEncoder

    # Encoding
    init_local_hidden = LocalEncoder.initHidden(batch_length)
    init_global_hidden = GlobalEncoder.initHidden(batch_length)
    local_encoder_outputs, local_hidden = LocalEncoder(inputs, init_local_hidden)
    global_input = initGlobalEncoderInput(MAX_BLOCK, batch_length, input_length,
                                          embedding_size, local_encoder_outputs)
    global_encoder_outputs, global_hidden = GlobalEncoder({"local_hidden_states":
                                                          global_input}, init_global_hidden)
    """
    Encoder Result Dimension: (batch, sequence length, hidden size)
    """
    local_encoder_outputs = local_encoder_outputs.permute(1, 0, 2)
    global_encoder_outputs = global_encoder_outputs.permute(1, 0, 2)

    # Debugging: Test encoder outputs
    # print(local_encoder_outputs)
    # print(global_encoder_outputs)

    # The decoder init for developing
    global_decoder = decoder.global_decoder
    local_decoder = decoder.local_decoder

    # Currently, we pad all box-scores to be the same length and blocks
    blocks_len = blocks_lens[0]

    # decoder starts
    gnh = global_decoder.initHidden(batch_length)
    lnh = local_decoder.initHidden(batch_length)

    # g_input = global_encoder_outputs[:, -1]
    g_input = global_decoder.initHidden(batch_length).permute(1, 2, 0)[:, :, -1]
    gnh = global_hidden
    l_input = Variable(torch.LongTensor(batch_length).zero_(), requires_grad=False)
    l_input = l_input.cuda() if use_cuda else l_input

    # Reshape the local_encoder outputs to (batch * blocks, blk_size, hidden)
    local_encoder_outputs = local_encoder_outputs.contiguous().view(batch_length * len(blocks_len),
                                                                    input_length // len(blocks_len),
                                                                    embedding_size)

    decoder_attentions = torch.zeros(target_length, input_length)

    # Initialize the Beam
    # Each Beam cell contains [prob, route,gnh, lnh, g_input, g_attn_weight, atten]
    beams = [[0, [(SOS_TOKEN, 0)], gnh, lnh, g_input, None, decoder_attentions]]
    if local_decoder.copy:
        oov_idx = Variable(torch.LongTensor(orm.shape))    
        oovs = {'<KWN>': 0}
        oovs_id2word = {0:'<KWN>'}
        oovs_ctr = len(oovs)
        # for i in range(orm.shape[1]):
        #     print("{} {} {}".format(data[0][0][i] ,rm[0,i], orm[0,i]))
        for i in range(orm.shape[1]):
            w = data[0][0][i][2]
            if orm[0,i].item() == 3:
                if w not in oovs:
                    oovs[w] = oovs_ctr
                    oovs_id2word[oovs_ctr] = w
                    oovs_ctr += 1
                oov_idx[0,i] = oovs[w]
            else:
                oov_idx[0,i] = 0
        oov_idx = oov_idx.cuda() if use_cuda else oov_idx

    # For each step
    for di in range(target_length):

        # For each information in the beam
        q = PriorityQueue()
        for beam in beams:

            prob, route, gnh, lnh, g_input, g_attn_weights, atten = beam
            destination = len(route) - 1

            # Get the lastest predecition
            decoder_input, inp_type = route[-1]
            if inp_type == 1:
                decoder_input = 3 # UNK_TOKEN

            # If <EOS>, do not search for it
            if decoder_input == EOS_TOKEN:
                q.push(beam, prob)
                continue
            
            if di == 0 or decoder_input == BLK_TOKEN:
                g_output, gnh, g_context, g_attn_weights = global_decoder(
                    g_input, gnh, global_encoder_outputs)

            l_input = Variable(torch.LongTensor([decoder_input]), requires_grad=False)
            l_input = l_input.cuda() if use_cuda else l_input

            l_output, lnh, l_context, l_attn_weights, pgen = local_decoder(
                l_input, lnh, g_attn_weights, local_encoder_outputs, blocks_len)
            
            l_attn_weights = l_attn_weights.squeeze(1)
            bg_attn_weights = g_attn_weights.view(batch_length * len(blocks_len), -1)
            # print(l_attn_weights)
            # print(g_attn_weights)
            # print(bg_attn_weights)
            combine_attn_weights = l_attn_weights * bg_attn_weights
            combine_attn_weights = combine_attn_weights.view(batch_length, -1)
            # print(torch.sum(combine_attn_weights))

            if local_decoder.copy:
                prob_copy = Variable(torch.zeros(l_output.shape), requires_grad=False)
                prob_copy = prob_copy.cuda() if use_cuda else prob_copy

                # Now we had rm as (batch, input) and combine_attn_weights as (batch, input)
                # Add up to the pgen probability matrix
                prob_copy = prob_copy.scatter_add(1, orm, combine_attn_weights)
                
                prob_oov = Variable(torch.zeros([1, len(oovs)]))
                prob_oov = prob_oov.cuda() if use_cuda else prob_oov

                prob_oov = prob_oov.scatter_add(1, oov_idx, combine_attn_weights)
                print(g_attn_weights)
                # prob_oov = prob_oov.log()
                prob_ukn = 1 - prob_oov[0,0]
                prob_oov[:,0] = 0
                prob_copy[0,3] -= prob_ukn
                # print(prob_copy)
                prob_oov *= (1-pgen)
                prob_oov = prob_oov.log()

                l_output_new = (l_output.exp() + (1 - pgen) * prob_copy).log()
            else:
                l_output_new = l_output

            # Get the attention vector at each prediction
            atten[destination, :combine_attn_weights.shape[1]] = combine_attn_weights.data[0, :]

            # decode the word
            topv, topi = l_output_new.data.topk(beam_size)
            if local_decoder.copy:
                if prob_oov.shape[1] >= beam_size:
                    copy_topv, copy_topi = prob_oov.data.topk(beam_size)
                else:
                    copy_topv, copy_topi = prob_oov.data.topk(prob_oov.shape[1])

            for i in range(beam_size):
                p = topv[0][i]
                idp = topi[0][i]
                new_beam = [prob + p, route + [(idp, 0)], gnh, lnh, lnh[-1, :, :], g_attn_weights, atten]
                q.push(new_beam, new_beam[0].item())
            if local_decoder.copy:
                for i in range(copy_topv.shape[1]):
                    p = copy_topv[0,i]
                    idp = copy_topi[0,i]
                    if idp.item()==0:
                        continue
                    new_beam = [prob + p, route + [(oovs_id2word[idp.item()], 1)], gnh, lnh, lnh[-1, :, :], g_attn_weights, atten]
                    q.push(new_beam, new_beam[0].item())

        # Keep the highest K probability
        beams = [q.pop() for i in range(beam_size)]

        # If the highest one is finished, we take that.
        if beams[0][1][-1] == 1:
            break

    # Get decoded_words and decoder_attetntions
    # decoded_words = [langs['summary'].index2word[w.item()] for w in beams[0][1][1:]]
    decoded_words = []
    for wpair in beams[0][1][1:]:
        if wpair[1] == 1:
            decoded_words.append("__"+wpair[0]+"__")
        else:
            decoded_words.append(langs['summary'].index2word[wpair[0].item()])
    decoder_attentions = beams[0][6]
    return decoded_words, decoder_attentions[:len(decoded_words)]


def predictwords(rt, re, rm, orm, data, encoder, decoder, embedding_size, langs, oov_dict, beam_size):
    """The function will predict the sentecnes given boxscore.
    Encode the given box score, decode it to sentences, and then
    return the prediction and attention matrix.
    While decoding, beam search will be conducted with default beam_size as 1.
    """
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = 1000

    encoder_outputs = Variable(torch.zeros(batch_length, input_length, embedding_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # Encoding
    init_hidden = encoder.initHidden(batch_length)
    inputs = {"rt": rt, "re": re, "rm": rm}    
    encoder_outputs, encoder_hiddens = encoder(inputs, init_hidden)
    
    context_vec = encoder_outputs[-1, :, :]
    # context_vec: (batch_size, hidden_dim)
    encoder_outputs = encoder_outputs.permute(1,0,2)

    decoder_hidden = decoder.initHidden(batch_length)
    decoder_hidden[0, :, :] = context_vec  # might be zero
    decoder_input = Variable(torch.LongTensor(batch_length).zero_(), requires_grad=False)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Initialize the Beam
    # Each Beam cell contains [prob, route, decoder_hidden, atten]
    decoder_attentions = torch.zeros(target_length, input_length)
    beams = [[0, [(SOS_TOKEN, 0)], decoder_hidden, decoder_attentions]]
    # (id, 0/1) 0 means in lang('summary'), 1 means oov
    # For each step
    if decoder.copy:
        oov_idx = Variable(torch.LongTensor(orm.shape))    
        oovs = {'<KWN>': 0}
        oovs_id2word = {0:'<KWN>'}
        oovs_ctr = len(oovs)
        # for i in range(orm.shape[1]):
        #     print("{} {} {}".format(data[0][0][i] ,rm[0,i], orm[0,i]))
        for i in range(orm.shape[1]):
            w = data[0][0][i][2]
            if orm[0,i].item() == 3:
                if w not in oovs:
                    oovs[w] = oovs_ctr
                    oovs_id2word[oovs_ctr] = w
                    oovs_ctr += 1
                oov_idx[0,i] = oovs[w]
            else:
                oov_idx[0,i] = 0
        oov_idx = oov_idx.cuda() if use_cuda else oov_idx
        

    for di in range(target_length):

        # For each information in the beam
        q = PriorityQueue()
        for beam in beams:
            # print(beam)
            prob, route, decoder_hidden, atten = beam
            destination = len(route) - 1

            # Get the lastest predecition
            decoder_input, inp_type = route[-1]
            if inp_type == 1:
                decoder_input = 3 # UNK_TOKEN

            # If <EOS>, do not search for it
            if decoder_input == EOS_TOKEN:
                q.push(beam, prob)
                continue

            decoder_input = Variable(torch.LongTensor([decoder_input]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            # print("{} {}".format(decoder_input, inp_type))
            decoder_output, decoder_hidden, decoder_context, decoder_attention, pgen = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            if decoder.copy:
                prob_copy = Variable(torch.zeros(decoder_output.shape))
                prob_copy = prob_copy.cuda() if use_cuda else prob_copy

                decoder_attention = decoder_attention.squeeze(1)
                prob_copy = prob_copy.scatter_add(1, orm, decoder_attention)
                #for i in range(orm.shape[1]):
                #    print("{} {}".format(rm[0,i], orm[0,i]))
                prob_oov = Variable(torch.zeros([1, len(oovs)]))
                prob_oov = prob_oov.cuda() if use_cuda else prob_oov

                prob_oov = prob_oov.scatter_add(1, oov_idx, decoder_attention)
                # prob_oov = prob_oov.log()
                prob_ukn = 1 - prob_oov[0,0]
                prob_oov[:,0] = 0
                prob_copy[0,3] -= prob_ukn
                # print(prob_copy)
                prob_oov *= (1-pgen)
                prob_oov = prob_oov.log()

                decoder_output_new = (decoder_output.exp() + (1-pgen)*prob_copy).log()

                # print(torch.sum(decoder_output_new, 1))
                # print(torch.sum(prob_oov, 1))
                decoder_attention = decoder_attention.unsqueeze(1)
            else:
                decoder_output_new = decoder_output

            # Get the attention vector at each prediction
            atten[destination, :decoder_attention.shape[2]] = decoder_attention.data[0, 0, :]

            # decode the word
            # print(decoder_output)
            topv, topi = decoder_output_new.data.topk(beam_size)
            if decoder.copy:
                if prob_oov.shape[1] >= beam_size:
                    copy_topv, copy_topi = prob_oov.data.topk(beam_size)
                else:
                    copy_topv, copy_topi = prob_oov.data.topk(prob_oov.shape[1])
                #print(copy_topv)
                #print(copy_topi)
            
            for i in range(beam_size):
                p = topv[0,i]
                idp = topi[0,i]
                if idp.item()<0 or idp.item()>=langs['summary'].n_words:
                    continue
                new_beam = [prob + p, route + [(idp, 0)], decoder_hidden, atten]
                q.push(new_beam, new_beam[0])
            if decoder.copy:
                for i in range(copy_topv.shape[1]):
                    p = copy_topv[0,i]
                    idp = copy_topi[0,i]
                    if idp.item()==0:
                        continue
                    new_beam = [prob + p, route + [(oovs_id2word[idp.item()], 1)], decoder_hidden, atten]
                    q.push(new_beam, new_beam[0].item())

        # Keep the highest K probability
        beams = [q.pop() for i in range(beam_size)]

        # If the highest one is finished, we take that.
        if beams[0][1][-1] == 1:
            break

    # Get decoded_words and decoder_attetntions
    #decoded_words = [langs['summary'].index2word[w.item()] for w in beams[0][1][1:]]
    decoded_words = []
    for wpair in beams[0][1][1:]:
        if wpair[1] == 1:
            decoded_words.append("__"+wpair[0]+"__")
        else:
            decoded_words.append(langs['summary'].index2word[wpair[0].item()])
    decoder_attentions = beams[0][3]
    return decoded_words, decoder_attentions[:len(decoded_words)]


def evaluate(valid_set, langs, embedding_size,
             encoder_style, decoder_style, 
             beam_size=1, verbose=False, oov_dict=None):
    """The evaluate procedure."""

    encoder, decoder, _, _ = model_initialization(encoder_style, 
                                            decoder_style, langs, 
                                            embedding_size, 0, pretrain=PRETRAIN, 
                                            layer_depth=LAYER_DEPTH, to_copy=TOCOPY, 
                                            iter_num=iterNum, load_optim=False)

    batch_size=1

    if decoder_style == 'HierarchicalRNN':
        decode_func = hierarchical_predictwords
    else:
        decode_func = predictwords
    # No need to calculate the loss
    criterion = None

    # Build the model
    model = Seq2Seq(encoder, decoder, None, decode_func, criterion, embedding_size, langs, oov_dict)
    model.eval()
    # Get evaluate data
    valid_iter = data_iter(valid_set, batch_size=1, shuffle=False)

    iteration = 0
    for dt in valid_iter:

        # Get data
        iteration += 1
        data, idx_data = get_batch(dt)
        rt, re, rm, orm, summary, osummary = idx_data

        # Debugging: check the input triplets
        # show_triplets(data[0][0])

        # Add paddings
        rt = addpaddings(rt)
        re = addpaddings(re)
        rm = addpaddings(rm)
        orm = addpaddings(orm)
        

        if decoder_style == 'HierarchicalRNN' and batch_size != 1:
            summary, osummary = add_sentence_paddings(summary, osummary)
        else:
            summary, data = addpaddings(summary, company=data)
            #osummary = addpaddings(osummary, to=6)

        rt = Variable(torch.LongTensor(rt), requires_grad=False)
        re = Variable(torch.LongTensor(re), requires_grad=False)
        rm = Variable(torch.LongTensor(rm), requires_grad=False)
        orm = Variable(torch.LongTensor(orm), requires_grad=False)

        # For Decoding
        summary = Variable(torch.LongTensor(summary), requires_grad=False)
        # osummary = Variable(torch.LongTensor(osummary), requires_grad=False)

        if use_cuda:
                rt, re, rm, orm, summary = rt.cuda(), re.cuda(), rm.cuda(), orm.cuda(), summary.cuda()

        # Get decoding words and attention matrix
        with torch.no_grad():
            decoded_words, decoder_attentions = model.seq_decode(rt, re, rm, orm, data, beam_size)

        res = ' '.join([ w for w in decoded_words[:-1] if w!='<PAD>'])
        # res = ' '.join(decoded_words[:-1])
        if verbose:
            print("Generate Summary {}:".format(iteration))
            print(res)

        # # FOR WRITING REPORTS ONLY
        # # Compare to the origin data
            print("Reference Summary:")
            triplets, gold_summary = data[0]

            for word in gold_summary:
                print(word, end=' ')
            print(' ')
            
            block_num = 0
            ctr = 0
            # count block
            while ctr < decoder_attentions.shape[1]:
                ctr_end = ctr
                while ctr_end < rt.shape[1] and ctr_end < decoder_attentions.shape[1] and rt[0, ctr_end] != EOB_TOKEN :
                    ctr_end+=1
                block_num += 1
                if ctr_end >= rt.shape[1]:
                    break
                ctr = ctr_end+1
            print(block_num)
            fig = plt.figure(figsize=(40,60))
            i = 0
            ctr = 0
            while ctr < decoder_attentions.shape[1]:
                ctr_end = ctr
                while ctr_end < rt.shape[1] and ctr_end < decoder_attentions.shape[1] and rt[0,ctr_end] != EOB_TOKEN :
                    ctr_end +=1
                ax = fig.add_subplot(block_num, 1 ,i+1)
                mat = ax.matshow(decoder_attentions.t()[ ctr: ctr_end+1 ,:], interpolation='nearest')
                ctr = ctr_end+1
                i+=1
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(mat, cax=cbar_ax)

            #ax.set_xticklabels(decoded_words)
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            # ax.set_yticklabels(['']+alpha)

            plt.savefig("./figures/"+PRETRAIN+'_'+str(iteration)+'.png')
        yield res

        

        # showAttention(triplets, decoded_words, decoder_attentions)
    return

def main():
    print("Start Evaluating")

    # Default parameter
    embedding_size = EMBEDDING_SIZE
    encoder_style = ENCODER_STYLE
    decoder_style = DECODER_STYLE

    # Prepare data for loading the model
    _, train_lang = loaddata(file_loc, 'train')

    # Load data for evaluation
    valid_data, _ = loaddata(file_loc, 'valid')
    valid_data, oov_dict = data2index(valid_data, train_lang)
    text_generator = evaluate(valid_data, train_lang, embedding_size,
                              encoder_style, decoder_style,
                              beam_size=10, verbose=True, oov_dict=oov_dict)

    # Generate Text
    start = time.time()
    for idx, text in enumerate(text_generator):
        print('Time: {}:\n'.format(gettime(start)))

if __name__ == '__main__':
    with torch.no_grad():
        main()
