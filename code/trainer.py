import torch
import torch.nn as nn
import torch.optim as optim
from typing import *
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import math
import glob
import logging
from utils import AverageMeter
import yaml
import wandb

from utils import translate_seq, cleanse_sent, calculate_metric


class Trainer:
    def __init__(self, hparams, loaders, model):
        self.hparams = hparams

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

        # dataloader
        self.train_iterator, self.valid_iterator, self.test_iterator = loaders

        # metric
        self.criterion = nn.CrossEntropyLoss(ignore_index=hparams.pad_idx)
        self.gradient_accumulation_step = self.hparams.gradient_accumulation_step

        self.step_total = (
            len(self.train_iterator) // self.gradient_accumulation_step * self.hparams.epoch
        )
        
        # model saving options
        self.version = 0
        while True:
            self.save_path = os.path.join(
                hparams.ckpt_path, f"version-{self.version}"
            )
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        if hparams.use_transformer:
            wandb.init(project=hparams.experiment_name, name=f'Transformer-{self.version}')
        else:
            wandb.init(project=hparams.experiment_name, name=f'{hparams.rnn_cell_type}-{self.version}')
        wandb.config.update(hparams)

        self.global_val_loss = float("inf")

        # save hyperparameters
        with open(
                os.path.join(self.save_path, "hparams.yaml"), "w", encoding="utf8"
        ) as outfile:
            yaml.dump(
                hparams, outfile, default_flow_style=False, allow_unicode=True
            )

        self.global_step = 0
        self.eval_step = (
            int(self.step_total * hparams.eval_ratio)
            if hparams.eval_ratio > 0
            else self.step_total // hparams.epoch
        )
        # optimizer, scheduler
        self.optimizer, self.scheduler = self.configure_optimizers()

        # experiment logging options
        self.best_result = {"version": self.version}
        self.log_step = hparams.log_step

        logging.basicConfig(
            filename=os.path.join(self.save_path, "experiment.log"),
            level=logging.INFO,
            format="%(asctime)s > %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %Z",
        )
        logging.info(
            f"[SCHEDULER] Total_step: {self.step_total} | Warmup step: {self.warmup_steps} | Accumulation step: {self.gradient_accumulation_step}"
        )


    def configure_optimizers(self):
        # optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # lr warmup scheduler
        self.warmup_steps = math.ceil(self.step_total * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.step_total,
        )

        return optimizer, scheduler

    def save_checkpoint(self, epoch: int, val_loss: float, model: nn.Module) -> None:
        logging.info(
            f"Val loss decreased ({self.global_val_loss:.4f} → {val_loss:.4f}). Saving model ..."
        )

        # save_path
        new_path = os.path.join(
            self.save_path, f"best_model_step_{self.global_step}_loss_{val_loss:.4f}.pt"
        )

        # remove old checkpoint
        for filename in glob.glob(os.path.join(self.save_path, "*.pt")):
            os.remove(filename)

        torch.save(model.state_dict(), new_path)
        self.global_val_loss = val_loss

    def fit(self) -> dict:
        self.optimizer.zero_grad()
        self.optimizer.step()

        for epoch in tqdm(range(self.hparams.epoch), desc="epoch"):
            self._train_epoch(epoch)

        return self.version

    def _train_epoch(self, epoch: int) -> None:
        train_loss = AverageMeter()
        
        for step, batch in tqdm(
            enumerate(self.train_iterator),
            desc="training steps",
            total=len(self.train_iterator),
        ):
            self.model.train()
            src, target = batch
            src, target = src.to(self.device), target.to(self.device)

            if self.hparams.use_transformer:
                output, _ = self.model(src, target[:,:-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt = target[:,1:].contiguous().view(-1)

            else:
                output_ = self.model(
                src, target, teacher_forcing_ratio=self.hparams.teacher_forcing_ratio
            )
                output_dim = output_.shape[-1]
                output = output_[1:].view(-1, output_dim).to(self.hparams.device)
                tgt = target[1:].view(-1)
                
            # tgt = [(tgt len - 1) * batch size]
            # output = [(tgt len - 1) * batch size, output dim]
            
            # compute loss
            loss = self.criterion(output, tgt)
            loss = loss / self.gradient_accumulation_step
            loss.backward()

            if (step+1) % self.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.hparams.clip_param
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
            train_loss.update(loss.item())  

            # step for evaluation
            # print(f'self.eval_step: {self.eval_step}')
            # print(f'self.global_step: {self.global_step}')

            if (self.global_step + 1) % self.eval_step == 0:
                val_loss = self.validate(epoch)
                logging.info(
                        f"[VAL] global step: {self.global_step} | val loss: {val_loss:.4f}"
                    )
                if val_loss < self.global_val_loss:
                    self.save_checkpoint(epoch, val_loss, self.model)
                wandb.log({'val_loss': val_loss})

            # step for logging
            if self.global_step != 0 and self.global_step % self.log_step == 0:
                logging.info(
                    f"[TRN] Version: {self.version} | Epoch: {epoch} | Global step: {self.global_step} | Train loss: {loss.item():.3f} | LR: {self.optimizer.param_groups[0]['lr']:.5f}"
                )
                wandb.log({'train_loss': loss.item()})

    def validate(self, epoch: int):
        self.model.eval()
        val_loss = AverageMeter()

        for step, batch in tqdm(
            enumerate(self.valid_iterator),
            desc="validation steps",
            total=len(self.valid_iterator),
        ):
            src, tgt = batch
            src, tgt = src.to(self.device), tgt.to(self.device)

            if self.hparams.use_transformer:
                output, _ = self.model(src, tgt[:,:-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt = tgt[:,1:].contiguous().view(-1)
            else:
                output_ = self.model(
                src, tgt, teacher_forcing_ratio=self.hparams.teacher_forcing_ratio
            )
                output_dim = output_.shape[-1]
                output = output_[1:].view(-1, output_dim).to(self.hparams.device)
                tgt = tgt[1:].view(-1)

            # tgt = [(tgt len - 1) * batch size]
            # output = [(tgt len - 1) * batch size, output dim]

            loss = self.criterion(output, tgt)

            val_loss.update(loss.item())
        return val_loss.avg

    def test(self, state_dict):
        total_output = []
        total_target = []

        test_loss = AverageMeter()

        self.model.load_state_dict(state_dict)
        self.model.eval()

        metric = {'total_bleu':0,
         'Rouge-L-P':0,
         'Rouge-L-R':0,
         'Rouge-L-F':0,
         'Rouge-2-P':0,
         'Rouge-2-R':0,
         'Rouge-2-F':0}
        cnt = 0
        for step, batch in tqdm(
            enumerate(self.test_iterator), desc="test step", total=len(self.test_iterator)
        ):
            print(f'step: {step}')
            src, target = batch
            src, target = src.to(self.device), target.to(self.device)

            if self.hparams.use_transformer:
                output, _ = self.model(src, target[:,:-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim).to(self.hparams.device)
                tgt = target[:,1:].contiguous().view(-1)
                
            else:
                output_ = self.model(
                src, target, teacher_forcing_ratio=self.hparams.teacher_forcing_ratio
            )
                output_dim = output_.shape[-1]
                output = output_[1:].view(-1, output_dim).to(self.hparams.device)
                tgt = target[1:].view(-1)
            # tgt = [(tgt len - 1) * batch size]
            # output = [(tgt len - 1) * batch size, output dim]

            loss = self.criterion(output, tgt)
            test_loss.update(loss.item())

            if self.hparams.use_transformer:
                pass
                # needs to be fixed

            else:
                import IPython; IPython.embed(); exit(1)
                for batch_output, batch_target in zip(total_output, total_target):
                    for idx in range(len(batch_output)):
                        cnt += 1
                        one_output, one_target = batch_output[:, idx], batch_target[:, idx]
                        
                        # prediction = translate_seq(one_output, self.model, self.hparams)
                        rouge, bleu = calculate_metric(one_output, cleanse_sent(one_target), self.model, self.hparams)
                        for key in rouge:
                            metric[key] += rouge[key]
                        metric['total_bleu'] += float(bleu)
                        
                        # if cnt % 100 == 0:
                        #     print(f"BLEU: {metric['total_bleu']}")
                        #     print(f"Rouge-L-P: {metric['Rouge-L-P']}")
        
                for key in metric:
                    metric[key] = metric[key] / cnt
                print(metric)

        wandb.log({'test_loss': test_loss.avg})

        logging.info(f"[TST] Test Loss: {test_loss.avg:.4f}")
        return {"test_loss": test_loss.avg}