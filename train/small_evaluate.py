"""Evaluate the model."""
import torch
from torch.autograd import Variable

from preprocessing import data_iter
from dataprepare import loaddata, data2index
from train import get_batch, model_initialization, addpaddings


from model import docEmbedding, Seq2Seq
from model import EncoderLIN, EncoderBiLSTM, EncoderBiLSTMMaxPool
from model import HierarchicalEncoderRNN, HierarchicalBiLSTM, HierarchicalLIN
from model import AttnDecoderRNN, HierarchicalDecoder
from util import PriorityQueue


from settings import file_loc, use_cuda
from settings import EMBEDDING_SIZE, ENCODER_STYLE, DECODER_STYLE
from settings import USE_MODEL

from util import load_model

# TODO move token numbering to a common file
SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2
EOB_TOKEN = 4
BLK_TOKEN = 5

def hierarchical_predictwords(rt, re, rm, encoder, decoder, embedding_size,
                              encoder_style, beam_size):
    """The function will predict the sentecnes given boxscore.

    Encode the given box score, decode it to sentences, and then
    return the prediction and attention matrix.

    While decoding, beam search will be conducted with default beam_size as 1.

    """
    batch_length = rt.size()[0]
    input_length = rt.size()[1]
    target_length = 1000

    MAX_BLOCK, blocks_lens = find_max_block_numbers(batch_length, langs, rm)
    BLOCK_JUMPS = 32

    LocalEncoder = encoder.LocalEncoder
    GlobalEncoder = encoder.GlobalEncoder

    # For now, these are redundant
    local_encoder_outputs = Variable(torch.zeros(batch_length, input_length, embedding_size))
    local_encoder_outputs = local_encoder_outputs.cuda() if use_cuda else local_encoder_outputs
    global_encoder_outputs = Variable(torch.zeros(batch_length, MAX_BLOCK, embedding_size))
    global_encoder_outputs = global_encoder_outputs.cuda() if use_cuda else global_encoder_outputs


    # Encoding
    if encoder_style == 'BiLSTM':
        init_hidden = encoder.initHidden(batch_length)
        encoder_hidden, encoder_hiddens = encoder(rt, re, rm, init_hidden)

        # Store memory information
        for ei in range(input_length):
            encoder_outputs[:, ei] = encoder_hiddens[:, ei]

    else:
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

    decoder_attentions = torch.zeros(target_length, input_length)

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
    beams = [[0, [SOS_TOKEN], decoder_hidden, decoder_attentions]]

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

            decoder_input = Variable(torch.LongTensor([decoder_input]), requires_grad=False)
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            decoder_output, decoder_hidden, decoder_context, decoder_attention, pgen = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            if decoder.copy:
                prob = Variable(torch.zeros(decoder_output.shape), requires_grad=False)
                prob = prob.cuda() if use_cuda else prob

                decoder_attention = decoder_attention.squeeze(1)
                prob = prob.scatter_add(1, rm, decoder_attention)

                decoder_output_new = (decoder_output.exp() + (1-pgen)*prob).log()
                decoder_attention = decoder_attention.unsqueeze(1)                
            else:
                decoder_output_new = decoder_output

            # Get the attention vector at each prediction
            atten[destination, :decoder_attention.shape[2]] = decoder_attention.data[0, 0, :]

            # decode the word
            # print(decoder_output)
            topv, topi = decoder_output.data.topk(beam_size)

            for i in range(beam_size):
                p = topv[0][i]
                idp = topi[0][i]
                new_beam = [prob + p, route + [idp], decoder_hidden, atten]
                # print(new_beam[0])
                q.push(new_beam, new_beam[0])

        # Keep the highest K probability
        beams = [q.pop() for i in range(beam_size)]

        # If the highest one is finished, we take that.
        if beams[0][1][-1] == 1:
            break

    # Get decoded_words and decoder_attetntions
    decoded_words = [lang.index2word[w.item()] for w in beams[0][1][1:]]
    decoder_attentions = beams[0][3]
    return decoded_words, decoder_attentions[:len(decoded_words)]


def evaluate(valid_set, langs, embedding_size,
             encoder_style, decoder_style, 
             use_model, beam_size=1, verbose=False):
    """The evaluate procedure."""

    encoder, decoder, _, _ = model_initialization(encoder_style, 
                                            decoder_style, langs, 
                                            embedding_size, 0, use_model)

    if decoder_style == 'HierarchicalRNN':
        decode_func = hierarchical_predictwords
    else:
        decode_func = predictwords
    # No need to calculate the loss
    criterion = None

    # Build the model
    model = Seq2Seq(encoder, decoder, None, decode_func, criterion, embedding_size, langs)

    # Get evaluate data
    valid_iter = data_iter(valid_set, batch_size=1, shuffle=False)

    iteration = 0
    for dt in valid_iter:

        # Get data
        iteration += 1
        data, idx_data = get_batch(dt)
        rt, re, rm, summary = idx_data

        # Debugging: check the input triplets
        # show_triplets(data[0][0])

        # Add paddings
        rt = addpaddings(rt)
        re = addpaddings(re)
        rm = addpaddings(rm)

        summary = addpaddings(summary)

        rt = Variable(torch.LongTensor(rt), requires_grad=False)
        re = Variable(torch.LongTensor(re), requires_grad=False)
        rm = Variable(torch.LongTensor(rm), requires_grad=False)

        # For Decoding
        summary = Variable(torch.LongTensor(summary), requires_grad=False)

        if use_cuda:
            rt, re, rm, summary = rt.cuda(), re.cuda(), rm.cuda(), summary.cuda()

        # Get decoding words and attention matrix
        decoded_words, decoder_attentions = predictwords(rt, re, rm, summary,
                                                         encoder, decoder, langs['summary'],
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
    return

def main():
    print("Start Evaluating")

    # Default parameter
    embedding_size = EMBEDDING_SIZE
    encoder_style = ENCODER_STYLE
    decoder_style = DECODER_STYLE
    use_model = USE_MODEL

    # Prepare data for loading the model
    train_data, train_lang = loaddata(file_loc, 'train')

    # Load data for evaluation
    valid_data, _ = loaddata(file_loc, 'valid')
    valid_data = data2index(valid_data, train_lang)
    text_generator = evaluate(valid_data, train_lang, embedding_size,
                              encoder_style, decoder_style,
                              use_model, beam_size=1, verbose=False)

    # Generate Text
    for idx, text in enumerate(text_generator):
        print('Generate Summary {}:\n{}'.format(idx + 1, text))

if __name__ == '__main__':
    main()
