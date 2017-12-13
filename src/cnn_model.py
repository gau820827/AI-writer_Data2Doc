import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# CNN model for relation classification

class Conv_relation_extractor(nn.Module):
    def __init__(self, config):
        super(Conv_relation_extractor, self).__init__()
        # reading parameters from config
        word_embed_size = config.getint('CNN', 'word_embed_size')
        pos_embed_size  = config.getint('CNN', 'pos_embed_size')
        word_embed_dim  = config.getint('CNN', 'wrod_embed_dim')
        pos_embed_dim   = config.getint('CNN', 'pos_embed_dim')
        total_embed_dim = word_embed_size + (2 * pos_embed_dim)

        filter_num      = config.getint('CNN', 'filter_num')
        kernel_sizes    = config.get('CNN', 'kernel_sizes')
        kernel_sizes = kernel_sizes.split(' ')
        self.kernel_sizes = [int(x) for x in kernel_sizes]
        self.max_len = config.getint('CNN', 'max_len')
        free_layer_size = config.getint('CNN', 'free_layer_size')
        output_size     = config.getint('CNN', 'output_size')
        dropput         = config.getint('CNN', 'dropput')

        self.embed_word = nn.Embedding(word_embed_size, word_embed_dim)
        self.embed_pos  = nn.Embedding(pos_embed_size, pos_embed_dim)

        # adding conv filters
        for i in range(len(self.kernel_sizes)):
            conv = nn.Conv1d(1, filter_num, total_embed_dim * self.kernel_sizes[i], stride=total_embed_dim)
            setattr(self, f'conv_{i}', conv)

        self.dropout = nn.Dropout(args.dropout)
        self.linear1 = nn.Linear(len(self.kernel_sizes) * filter_num, free_layer_size)
        self.linear2 = nn.Linear(free_layer_size, output_size)

    def forward(self, inp):
        word_embed = self.embed_word(inp.word)
        pos_embed1 = self.embed_pos(inp.pos1)
        pos_embed2 = self.embed_pos(inp.pos2)
        x = torch.cat( (word_embed, pos_embed1, pos_embed2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.max_len - self.kernel_sizes[i] + 1)
                .view(-1, self.kernel_sizes[i])
            for i in range(len(self.kernel_sizes))]

        x = torch.cat(conv_results, 1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = nn.SoftMax(x)
        return x
