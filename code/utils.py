import torch
import torch.nn as nn
from ignite.metrics.nlp import Bleu
from ignite.metrics import Rouge


def init_weights(model, init_type):
    if init_type == "uniform":
        for name, param in model.named_parameters():
            nn.init.uniform_(param)
    elif init_type == "xavier":
        for name, param in model.named_parameters():
            nn.init.xavier_uniform_(param)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def translate_seq(sequence, model, hparams):
    src_tensor = torch.LongTensor(sequence).unsqueeze(0).to(hparams.device)
    src_tensor = sequence.unsqueeze(0).to(hparams.device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [hparams.SOS_token]

    for _ in range(hparams.tgt_maxlen):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(hparams.device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)
        if pred_token == hparams.EOS_token:
            break

    trg_indexes = cleanse_sent(trg_indexes)
    
    return trg_indexes

def cleanse_sent(sent):
    target_idx = -1
    for idx, s in enumerate(sent):
        if s == 1:
            target_idx = idx
            break

    return sent[1:target_idx]  

def calculate_metric(src_seq, tgt_seq, model, hparam):
    trgs = []
    pred_trgs = []
            
    pred_trg = translate_seq(src_seq, model, hparam)
    
    pred_trgs.append(pred_trg)
    trgs.append(tgt_seq.tolist())
    
    pred_trgs = [[str(k) for k in i] for i in pred_trgs]
    trgs = [[str(l) for l in j] for j in trgs]
    
    # Rouge Score
    m = Rouge(variants=["L", 2], multiref="average")
    m.update((pred_trgs, [trgs]))
    rouge_score = m.compute()
    
    # BLEU Score
    m = Bleu(ngram=1)
    m.update((pred_trgs, [trgs]))
    bleu_score = m.compute()

    return rouge_score, bleu_score

def calculate_metrics(src_seq, tgt_seq):
    src_seq = [str(i) for i in cleanse_sent(src_seq)]
    tgt_seq = [str(j) for j in cleanse_sent(tgt_seq)]

    m = Rouge(variants=["L", 2], multiref="average")
    m.update((src_seq, [tgt_seq]))
    rouge_score = m.compute()
    
    # BLEU Score
    m = Bleu(ngram=1)
    m.update((src_seq, [tgt_seq]))
    bleu_score = m.compute()

    return rouge_score, bleu_score