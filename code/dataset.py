import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import os
from typing import List, Set, Dict, Tuple

SOS_token = 0
EOS_token = 1
PAD_token = 2

def open_file(file_loc: str) -> List:
    file = open(file_loc, "r")
    file_lines = file.readlines()
    file_lines = [list(map(int, line.split())) for line in file_lines]
    return file_lines

class SeqDataset(Dataset):
    def __init__(self, src_seq: List, tgt_seq: List):
        self.src_seq = src_seq
        self.tgt_seq = tgt_seq

        self.src_maxlen = len(max(self.src_seq, key=len))
        self.tgt_maxlen = len(max(self.tgt_seq, key=len))

    def max_len_return(self) -> [int, int]:
        return self.src_maxlen, self.tgt_maxlen

    def __len__(self) -> int:
        return len(self.src_seq)

    def __getitem__(self, idx: int) -> List:
        src = torch.tensor(self.src_seq[idx])
        tgt = torch.tensor(self.tgt_seq[idx])

        return src, tgt


# def pad_collate(batch) -> (List, List):
#     (xs, ys) = zip(*batch)

#     x_pad = pad_sequence(xs, padding_value=2)
#     y_pad = pad_sequence(ys, padding_value=2)

#     return x_pad, y_pad

class Collator:
    def __init__(self, *params):
        self.params = params
    
    def __call__(self, batch):
        (xs, ys) = zip(*batch)
        x_pad = pad_sequence(xs, batch_first=self.params[0], padding_value=2)
        y_pad = pad_sequence(ys, batch_first=self.params[0], padding_value=2)

        return x_pad, y_pad

def get_loader(
    hparams
):
    train_source_loc, train_target_loc = os.path.join(
        hparams.root_path, "train_source.txt"
    ), os.path.join(hparams.root_path, "train_target.txt")
    test_source_loc, test_target_loc = os.path.join(
        hparams.root_path, "test_source.txt"
    ), os.path.join(hparams.root_path, "test_target.txt")

    train_source, train_target = open_file(train_source_loc), open_file(
        train_target_loc
    )
    test_source, test_target = open_file(test_source_loc), open_file(test_target_loc)

    special_vocabs = ["<sos>", "<eos>", "<pad>"]

    # generate vocabulary dictionary
    def get_vocab(train_src, train_tgt, test_src, test_tgt):
        train_src_vocab, train_tgt_vocab = set(
            [i for line in train_src for i in line]
        ), set([i for line in train_tgt for i in line])
        test_src_vocab, test_tgt_vocab = set(
            [i for line in test_src for i in line]
        ), set([i for line in test_tgt for i in line])

        # Combine vocab of (train source - test source) & (train target - test target)
        train_src_vocab |= test_src_vocab
        train_tgt_vocab |= test_tgt_vocab

        # Add '<sos>', '<eos>', '<pad>' to the vocab
        src_vocab_dict, tgt_vocab_dict = {}, {}
        for idx, value in enumerate(special_vocabs):
            src_vocab_dict[idx] = value
            tgt_vocab_dict[idx] = value
        src_vocab_dict = {
            value + len(special_vocabs): idx + len(special_vocabs)
            for idx, value in enumerate(sorted(train_src_vocab))
        }
        tgt_vocab_dict = {
            value + len(special_vocabs): idx + len(special_vocabs)
            for idx, value in enumerate(sorted(train_tgt_vocab))
        }

        return src_vocab_dict, tgt_vocab_dict

    src_vocab, tgt_vocab = get_vocab(
        train_source, train_target, test_source, test_target
    )

    train_source = [
        [0] + [src_vocab[i + len(special_vocabs)] for i in seq] + [1]
        for seq in train_source
    ]
    train_target = [
        [0] + [tgt_vocab[i + len(special_vocabs)] for i in seq] + [1]
        for seq in train_target
    ]

    test_source = [
        [0] + [src_vocab[i + len(special_vocabs)] for i in seq] + [1]
        for seq in test_source
    ]
    test_target = [
        [0] + [tgt_vocab[i + len(special_vocabs)] for i in seq] + [1]
        for seq in test_target
    ]

    train_len = int(len(train_source) * (1.0-hparams.valid_ratio))
    val_len = len(train_source) - train_len

    seq_dataset = SeqDataset(train_source, train_target)
    train_dataset, val_dataset = random_split(
        seq_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(hparams.seed)
    )
    train_src_maxlen, train_tgt_maxlen = seq_dataset.max_len_return()

    test_dataset = SeqDataset(test_source, test_target)

    test_src_maxlen, test_tgt_maxlen = test_dataset.max_len_return()

    src_maxlen, tgt_maxlen = 0,0
    if train_src_maxlen > test_src_maxlen:
        src_maxlen = train_src_maxlen
    else:
        src_maxlen = test_src_maxlen
    
    if train_tgt_maxlen > test_tgt_maxlen:
        tgt_maxlen = train_tgt_maxlen
    else:
        tgt_maxlen = test_tgt_maxlen

    # print(f'src_maxlen: {src_maxlen}')
    # print(f'tgt_maxlen: {tgt_maxlen}')

    collator = Collator(hparams.use_transformer)
    train_iterator = DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=collator
    )
    valid_iterator = DataLoader(
        val_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=collator
    )
    test_iterator = DataLoader(
        test_dataset, batch_size=hparams.batch_size, collate_fn=collator
    )


    return train_iterator, valid_iterator, test_iterator, src_vocab, tgt_vocab, src_maxlen, tgt_maxlen
