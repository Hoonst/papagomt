import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import numpy as np
from typing import List, Set, Dict, Tuple

import os
import random

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class Encoder(nn.Module):
    def __init__(self, input_dim, hparams):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = hparams.emb_dim
        self.hid_dim = hparams.hid_dim
        self.n_layers = hparams.n_layers
        self.dropout = hparams.dropout

        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)

        if hparams.rnn_cell_type == "rnn":
            self.rnn = nn.RNN(
                self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout
            )
        elif hparams.rnn_cell_type == "lstm":
            self.rnn = nn.LSTM(
                self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout
            )
        elif hparams.rnn_cell_type == "gru":
            self.rnn = nn.GRU(
                self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout
            )

        self.dropout_f = nn.Dropout(self.dropout)

    def forward(self, src):

        # src = [src len, batch size]

        embedded = self.dropout_f(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, hparams, attention):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = hparams.emb_dim
        self.hid_dim = hparams.hid_dim
        self.n_layers = hparams.n_layers
        self.dropout = hparams.dropout
        self.use_attention = hparams.use_attention
        self.attention = attention
        self.attn_combine = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.output_combine = nn.Linear(self.hid_dim * 3, self.hid_dim)

        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)

        if hparams.rnn_cell_type == "rnn":
            self.rnn = nn.RNN(
                self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout
            )
        elif hparams.rnn_cell_type == "lstm":
            self.rnn = nn.LSTM(
                self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout
            )
        elif hparams.rnn_cell_type == "gru":
            self.rnn = nn.GRU(
                self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout
            )

        self.fc_out = nn.Linear(hparams.hid_dim, output_dim)
        self.dropout_f = nn.Dropout(self.dropout)

    def forward(self, input_seq, hidden, encoder_outputs):

        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input_seq = input_seq.unsqueeze(0)
        embedded = self.dropout_f(self.embedding(input_seq))

        if self.use_attention:

            a = self.attention(hidden, encoder_outputs)
            a = a.unsqueeze(1)

            encoder_outputs = encoder_outputs.permute(1,0,2)

            weighted = torch.bmm(a, encoder_outputs)

            weighted = weighted.permute(1, 0, 2)
            input_seq = torch.cat((embedded, weighted), dim = 2)
            input_seq = self.attn_combine(input_seq)

            output, hidden = self.rnn(input_seq, hidden)
            
            # embedded = embedded.squeeze(0)
            # output = output.squeeze(0)
            # weighted = weighted.squeeze(0)
            
            # hidden = hidden.squeeze(0)
            prediction = self.fc_out(self.output_combine(torch.cat((output, weighted, embedded), dim=2)))
            prediction = prediction.squeeze(0)

        else:    
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc_out(output.squeeze(0))
        # embedded = [1, batch size, emb dim]

        #         hidden = tuple([_cat(h) for h in (hidden, cell)])
        #         print(hidden.shape)
        

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        

        # prediction = [batch size, output dim]

        return prediction, hidden

class Attention(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.emb_dim = hparams.emb_dim
        self.hid_dim = hparams.hid_dim
        self.n_layers = hparams.n_layers
        self.dropout = hparams.dropout
        
        self.attn = nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times

        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, hparams):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.use_attention = hparams.use_attention

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input_seq = trg[0, :]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input_seq, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_seq = trg[t] if teacher_force else top1
        return outputs
