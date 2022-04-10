from config import load_config
from dataset import get_loader
# from model import Encoder, Decoder, Seq2Seq, Attention
from utils import init_weights
from trainer import Trainer
import torch
import torch.nn as nn

import os
import glob

def main(hparams):
    train_iterator, valid_iterator, test_iterator, src_vocab, tgt_vocab, src_maxlen, tgt_maxlen = get_loader(
        hparams
    )
    data_loaders = (train_iterator, valid_iterator, test_iterator)
    vocabs = (src_vocab, tgt_vocab)
    
    special = 3
    src_vocab_len = len(src_vocab) + special
    tgt_vocab_len = len(tgt_vocab) + special

    length_spare = 5
    hparams.src_maxlen = src_maxlen + length_spare
    hparams.tgt_maxlen = tgt_maxlen + length_spare

    hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams.SOS_token, hparams.EOS_token, hparams.PAD_token = 0, 1, 2

    if hparams.use_transformer:
        from transformer import Encoder, Decoder, Seq2Seq
        encoder = Encoder(src_vocab_len, hparams)
        decoder = Decoder(tgt_vocab_len, hparams)
        model = Seq2Seq(encoder, decoder, hparams.PAD_token, hparams.PAD_token, hparams)

    else:
        from model import Encoder, Decoder, Seq2Seq, Attention
        attention = Attention(hparams)
        encoder = Encoder(src_vocab_len, hparams)
        decoder = Decoder(tgt_vocab_len, hparams, attention)

        model = Seq2Seq(encoder, decoder, hparams)

    # model.apply(lambda x: init_weights(x, hparams.init_type))

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)
    # for name, param in model.named_parameters():
    #     nn.init.xavier_uniform_(param)

    trainer = Trainer(hparams, data_loaders, model)
    best_result_version = trainer.fit()

    # if hparams.test:
    
    state_dict = torch.load(
        glob.glob(
            os.path.join(hparams.ckpt_path, f"version-{best_result_version}/best_model_*.pt")
        )[0],
    )
    print(f"version-{best_result_version}/best_model_*.pt")
    test_result = trainer.test(state_dict)
    print(f'test_result: {test_result}')
        


if __name__ == "__main__":
    hparams = load_config()
    main(hparams)
