import os
import math
import pickle
import argparse
import random
from collections import defaultdict
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from tqdm import tqdm, trange
from transformers import AutoTokenizer, BertModel, BertConfig

from module import Att2Seq, Mine
from utils import (
    rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens,
    unique_sentence_percent, feature_detect, feature_matching_ratio,
    feature_coverage_ratio, feature_diversity, WordDictionary
)


class MutualInformationEstimator:
    def __init__(self, mi_estimator: Mine, device: torch.device, ma_rate: float = 0.01):
        self.mi_estimator = mi_estimator
        self.device = device
        self.ma_rate = ma_rate
        self.ma_et = 1.0
        self.optimizer = torch.optim.Adam(mi_estimator.parameters(), lr=1e-4)

    def mutual_information(
        self,
        joint: torch.Tensor,
        marginal: torch.Tensor,
        batch: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.mi_estimator(joint)
        et = torch.exp(self.mi_estimator(marginal))
        if batch:
            mi_lb = t - torch.log(et)
        else:
            mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def train_step(
        self,
        joint: torch.Tensor,
        marginal: torch.Tensor
    ) -> float:
        joint = joint.to(self.device)
        marginal = marginal.to(self.device)
        
        self.optimizer.zero_grad()
        mi_lb, t, marginal_t = self.mutual_information(joint, marginal)
        
        self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * torch.mean(marginal_t).detach()
        
        loss = -(torch.mean(t) - (1 / self.ma_et) * torch.mean(marginal_t))
        loss.backward()
        self.optimizer.step()
        
        return mi_lb.detach().cpu().item()


class PolicyGradientTrainer:
    def __init__(
        self,
        model: Att2Seq,
        mi_estimator: Mine,
        ref_model: Att2Seq,
        tokenizer: AutoTokenizer,
        bert_encoder: BertModel,
        device: torch.device,
        text_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        args: argparse.Namespace,
        word_dict: WordDictionary
    ):
        self.model = model
        self.mi_estimator = mi_estimator
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.bert_encoder = bert_encoder
        self.device = device
        self.text_criterion = text_criterion
        self.optimizer = optimizer
        self.args = args
        self.word_dict = word_dict
        self.eos_idx = word_dict.word2idx['<eos>']
        self.pad_idx = word_dict.word2idx['<pad>']

    def freeze_model(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def activate_model(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = True

    def generate_text(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        rate: torch.Tensor,
        seq: torch.Tensor
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        batch_size = user.size(0)
        inputs = seq[:, :1].to(self.device)
        ids = inputs
        hidden = None
        hidden_c = None
        words_lens = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        gen_probs = []
        ref_probs = []
        
        for idx in range(self.args.words):
            if idx == 0:
                hidden = self.model.encoder(user, item, rate)
                hidden_c = torch.zeros_like(hidden)
                log_word_prob, hidden, hidden_c = self.model.decoder(inputs, hidden, hidden_c)
                if self.ref_model is not None:
                    ref_log_word_prob, _, _ = self.ref_model.decoder(inputs, hidden, hidden_c)
            else:
                log_word_prob, hidden, hidden_c = self.model.decoder(inputs, hidden, hidden_c)
                if self.ref_model is not None:
                    ref_log_word_prob, _, _ = self.ref_model.decoder(inputs, hidden, hidden_c)
            
            word_prob = log_word_prob.squeeze().exp()
            inputs = torch.multinomial(word_prob, 1)
            prob = word_prob.gather(1, inputs).squeeze()
            gen_probs.append(prob)
            
            if self.ref_model is not None:
                ref_prob = ref_log_word_prob.squeeze().exp().gather(1, inputs).squeeze()
                ref_probs.append(ref_prob)
            
            is_eos = (inputs == self.eos_idx).squeeze()
            not_end = words_lens == 0
            if idx != self.args.words - 1:
                words_lens[not_end & is_eos] = idx + 1
                if (words_lens != 0).all():
                    break
            else:
                words_lens[not_end] = self.args.words
            
            ids = torch.cat([ids, inputs], 1)
        
        idss = ids[:, 1:].tolist()
        tokens_predict = [ids2tokens(ids, self.word_dict) for ids in idss]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        
        gen_probs = torch.stack(gen_probs, dim=1)
        ref_probs = torch.stack(ref_probs, dim=1) if self.ref_model is not None else None
        
        return text_predict, words_lens, gen_probs, ref_probs

    def compute_bert_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_encoder(**inputs)
        
        return outputs.pooler_output

    def compute_mi_reward(
        self,
        embeddings: torch.Tensor,
        oht_ys: torch.Tensor,
        shuffled_oht_ys: torch.Tensor
    ) -> torch.Tensor:
        joint = torch.cat([embeddings, oht_ys.to(self.device)], dim=1)
        marginal = torch.cat([embeddings, shuffled_oht_ys.to(self.device)], dim=1)
        
        mi, _, _ = self.mi_estimator.mutual_information(joint, marginal, batch=True)
        return mi.squeeze()

    def train_step(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        rate: torch.Tensor,
        seq: torch.Tensor,
        feature_var: torch.Tensor,
        kl_weight: float,
        entropy_weight: float,
        mc_samples: int = 5
    ) -> Tuple[float, float]:
        batch_size = user.size(0)
        user = user.to(self.device)
        item = item.to(self.device)
        seq = seq.to(self.device)
        rate = rate.to(self.device)
        
        oht_ys = F.one_hot(rate - 1, num_classes=5)
        shuffled_rate = rate.cpu().numpy().copy()
        np.random.shuffle(shuffled_rate)
        shuffled_rate = torch.tensor(shuffled_rate, device=self.device)
        shuffled_oht_ys = F.one_hot(shuffled_rate - 1, num_classes=5)
        
        gen_log_probs_list = []
        ref_log_probs_list = []
        rewards_list = []
        mi_values = []
        
        for _ in range(mc_samples):
            text_predict, words_lens, gen_probs, ref_probs = self.generate_text(
                user, item, rate, seq
            )
            
            embeddings = self.compute_bert_embeddings(text_predict)
            mi_reward = self.compute_mi_reward(embeddings, oht_ys, shuffled_oht_ys)
            
            gen_log_probs = gen_probs.log()
            for i, l in enumerate(words_lens):
                if l < self.args.words:
                    gen_log_probs[i, l:] = 0
            
            gen_log_probs = gen_log_probs.sum(dim=1)
            gen_log_probs_list.append(gen_log_probs)
            
            if ref_probs is not None:
                ref_log_probs = ref_probs.log()
                for i, l in enumerate(words_lens):
                    if l < self.args.words:
                        ref_log_probs[i, l:] = 0
                ref_log_probs = ref_log_probs.sum(dim=1)
                ref_log_probs_list.append(ref_log_probs)
            
            rewards_list.append(mi_reward)
            mi_values.append(mi_reward.mean().item())
        
        gen_log_probs_tensor = torch.stack(gen_log_probs_list, dim=1)
        rewards_tensor = torch.stack(rewards_list, dim=1)
        
        if ref_log_probs_list:
            ref_log_probs_tensor = torch.stack(ref_log_probs_list, dim=1)
            kl_penalty = (gen_log_probs_tensor - ref_log_probs_tensor).detach()
            rewards_tensor = rewards_tensor - kl_weight * kl_penalty
        
        entropy_bonus = -gen_log_probs_tensor.detach()
        rewards_tensor = rewards_tensor + entropy_weight * entropy_bonus
        
        rewards_tensor = rewards_tensor - rewards_tensor.mean(dim=1, keepdim=True)
        pg_loss = -(rewards_tensor * gen_log_probs_tensor).mean()
        
        log_word_prob = self.model(user, item, rate, seq[:, :-1])
        nll_loss = self.text_criterion(
            log_word_prob.view(-1, len(self.word_dict)), 
            seq[:, 1:].reshape(-1)
        )
        
        total_loss = nll_loss + pg_loss
        avg_mi = np.mean(mi_values)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()
        
        return total_loss.item(), avg_mi

    def evaluate_step(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        rate: torch.Tensor,
        seq: torch.Tensor,
        feature_var: torch.Tensor,
        kl_weight: float,
        entropy_weight: float
    ) -> Tuple[float, float]:
        with torch.no_grad():
            return self.train_step(
                user, item, rate, seq, feature_var, 
                kl_weight, entropy_weight, mc_samples=1
            )


class TrainingOrchestrator:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.device = device
        self.corpus = None
        self.model = None
        self.tokenizer = None
        self.bert_encoder = None
        self.mi_trainer = None
        self.pg_trainer = None
        self.optimizer = None
        self.text_criterion = None
        self.best_metrics = {
            'val_loss': float('inf'),
            'mi': float('-inf'),
            'epoch': 0
        }

    def setup_directories(self):
        os.makedirs(self.args.checkpoint, exist_ok=True)

    def load_data(self):
        self.corpus = DataLoader(
            self.args.data_path, 
            self.args.index_dir, 
            self.args.vocab_size
        )
        self.train_data = Batchify(
            self.corpus.train, 
            self.corpus.word_dict, 
            self.args.words, 
            self.args.batch_size
        )
        self.val_data = Batchify(
            self.corpus.valid, 
            self.corpus.word_dict, 
            self.args.words, 
            self.args.batch_size
        )
        self.test_data = Batchify(
            self.corpus.test, 
            self.corpus.word_dict, 
            self.args.words, 
            self.args.batch_size
        )

    def initialize_models(self):
        nuser = len(self.corpus.user_dict)
        nitem = len(self.corpus.item_dict)
        ntoken = len(self.corpus.word_dict)
        
        self.model = Att2Seq(
            nuser, nitem, 5, ntoken, 
            self.args.emsize, self.args.nhid,
            self.args.dropout, self.args.nlayers
        ).to(self.device)
        
        # Load pretrained models
        self.model.load_state_dict(torch.load('../Att2Seq/model.pt'))
        self.model.ref_model = torch.load('../Att2Seq/model.pt').to(self.device)
        self.model.mi_estimator = Mine(input_size=773).to(self.device)
        self.model.mi_estimator.load_state_dict(
            torch.load('../MINE/checkpoints/tripadvisor_review_r_mi')['model_state_dict']
        )

    def initialize_bert(self):
        self.tokenizer = AutoTokenizer.from_pretrained("../bert_models/my_tripadvisor_model")
        config = BertConfig.from_pretrained("../bert_models/my_tripadvisor_model")
        self.bert_encoder = BertModel(config).to(self.device)
        self.bert_encoder.load_state_dict(
            torch.load("../bert_models/my_tripadvisor_model/pytorch_model.bin")
        )
        
        # Freeze BERT
        for param in self.bert_encoder.parameters():
            param.requires_grad = False

    def setup_training_components(self):
        self.text_criterion = nn.NLLLoss(ignore_index=self.corpus.word_dict.word2idx['<pad>'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        self.mi_trainer = MutualInformationEstimator(
            self.model.mi_estimator, self.device
        )
        
        self.pg_trainer = PolicyGradientTrainer(
            model=self.model,
            mi_estimator=self.model.mi_estimator,
            ref_model=self.model.ref_model,
            tokenizer=self.tokenizer,
            bert_encoder=self.bert_encoder,
            device=self.device,
            text_criterion=self.text_criterion,
            optimizer=self.optimizer,
            args=self.args,
            word_dict=self.corpus.word_dict
        )

    def train_mi_estimator(self, data: Batchify) -> float:
        self.model.mi_estimator.train()
        mi_values = []
        
        while True:
            user, item, rate, seq, feature_var = data.next_batch()
            oht_ys = F.one_hot(rate - 1, num_classes=5)
            
            shuffled_rate = rate.numpy().copy()
            np.random.shuffle(shuffled_rate)
            shuffled_oht_ys = F.one_hot(torch.tensor(shuffled_rate) - 1, num_classes=5)
            
            inputs = seq[:, :1].to(self.device)
            hidden = self.model.encoder(user.to(self.device), item.to(self.device), rate.to(self.device))
            hidden_c = torch.zeros_like(hidden)
            
            ids = inputs
            for idx in range(self.args.words):
                if idx == 0:
                    log_word_prob, hidden, hidden_c = self.model.decoder(inputs, hidden, hidden_c)
                else:
                    log_word_prob, hidden, hidden_c = self.model.decoder(inputs, hidden, hidden_c)
                
                word_prob = log_word_prob.squeeze().exp()
                inputs = torch.argmax(word_prob, dim=1, keepdim=True)
                ids = torch.cat([ids, inputs], 1)
            
            idss = ids[:, 1:].tolist()
            tokens_predict = [ids2tokens(ids, self.corpus.word_dict) for ids in idss]
            text_predict = [' '.join(tokens) for tokens in tokens_predict]
            
            embeddings = self.pg_trainer.compute_bert_embeddings(text_predict)
            
            joint = torch.cat([embeddings.cpu(), oht_ys], dim=1)
            marginal = torch.cat([embeddings.cpu(), shuffled_oht_ys], dim=1)
            
            mi_value = self.mi_trainer.train_step(joint, marginal)
            mi_values.append(mi_value)
            
            if data.step == data.total_step:
                break
        
        return np.mean(mi_values)

    def train_generator(
        self, 
        data: Batchify,
        kl_weight: float,
        entropy_weight: float
    ) -> Tuple[float, float]:
        self.model.train()
        losses = []
        mi_values = []
        
        while True:
            user, item, rate, seq, feature_var = data.next_batch()
            loss, mi = self.pg_trainer.train_step(
                user, item, rate, seq, feature_var,
                kl_weight, entropy_weight
            )
            
            losses.append(loss)
            mi_values.append(mi)
            
            if data.step == data.total_step:
                break
        
        return np.mean(losses), np.mean(mi_values)

    def evaluate_generator(
        self, 
        data: Batchify,
        kl_weight: float,
        entropy_weight: float
    ) -> Tuple[float, float]:
        self.model.eval()
        losses = []
        mi_values = []
        
        while True:
            user, item, rate, seq, feature_var = data.next_batch()
            loss, mi = self.pg_trainer.evaluate_step(
                user, item, rate, seq, feature_var,
                kl_weight, entropy_weight
            )
            
            losses.append(loss)
            mi_values.append(mi)
            
            if data.step == data.total_step:
                break
        
        return np.mean(losses), np.mean(mi_values)

    def run_training(
        self,
        kl_weights: List[float],
        entropy_weights: List[float]
    ):
        endure_count = 0
        
        for kl_weight in kl_weights:
            for entropy_weight in entropy_weights:
                print(f'Training with KL weight: {kl_weight}, Entropy weight: {entropy_weight}')
                
                for epoch in range(1, self.args.epochs + 1):
                    print(f'{now_time()} Epoch {epoch}')
                    
                    if epoch % 2 == 0:
                        print('Training MI estimator...')
                        self.pg_trainer.freeze_model(self.model)
                        self.pg_trainer.activate_model(self.model.mi_estimator)
                        
                        train_mi = self.train_mi_estimator(self.train_data)
                        val_mi = self.train_mi_estimator(self.val_data)  # Note: Should be evaluate_mi_estimator
                        
                        print(f'{now_time()} Train MI: {train_mi:.4f}, Val MI: {val_mi:.4f}')
                        
                        torch.save(
                            self.model.mi_estimator.state_dict(),
                            f'tripadvisor_backbone_with_r/r_mi_estimator_kl_{kl_weight}_entropy_{entropy_weight}_mmi_r.pt'
                        )
                    else:
                        print('Training generator...')
                        self.pg_trainer.freeze_model(self.model.mi_estimator)
                        self.pg_trainer.activate_model(self.model)
                        
                        train_loss, train_mi = self.train_generator(
                            self.train_data, kl_weight, entropy_weight
                        )
                        val_loss, val_mi = self.evaluate_generator(
                            self.val_data, kl_weight, entropy_weight
                        )
                        
                        print(f'{now_time()} Train Loss: {train_loss:.4f}, MI: {train_mi:.4f}')
                        print(f'{now_time()} Val Loss: {val_loss:.4f}, MI: {val_mi:.4f}')
                        
                        if val_mi > self.best_metrics['mi'] or val_loss < self.best_metrics['val_loss']:
                            if val_mi > self.best_metrics['mi']:
                                self.best_metrics['mi'] = val_mi
                            if val_loss < self.best_metrics['val_loss']:
                                self.best_metrics['val_loss'] = val_loss
                            
                            self.best_metrics['epoch'] = epoch
                            torch.save(
                                self.model.state_dict(),
                                f'tripadvisor_backbone_with_r/pg_kl_{kl_weight}_entropy_{entropy_weight}_mmi_r_{epoch}.pt'
                            )
                            endure_count = 0
                        else:
                            endure_count += 1
                            if endure_count >= self.args.endure_times:
                                print(f'Early stopping at epoch {epoch}')
                                break

    def generate_predictions(self, data: Batchify) -> Tuple[List[List[str]], List[int]]:
        self.model.eval()
        idss_predict = []
        ratings = []
        
        with torch.no_grad():
            while True:
                user, item, rate, seq, _ = data.next_batch()
                user = user.to(self.device)
                item = item.to(self.device)
                ratings.extend(rate.tolist())
                rate = rate.to(self.device)
                
                inputs = seq[:, :1].to(self.device)
                hidden = self.model.encoder(user, item, rate)
                hidden_c = torch.zeros_like(hidden)
                
                ids = inputs
                for idx in range(self.args.words):
                    if idx == 0:
                        log_word_prob, hidden, hidden_c = self.model.decoder(inputs, hidden, hidden_c)
                    else:
                        log_word_prob, hidden, hidden_c = self.model.decoder(inputs, hidden, hidden_c)
                    
                    word_prob = log_word_prob.squeeze().exp()
                    inputs = torch.argmax(word_prob, dim=1, keepdim=True)
                    ids = torch.cat([ids, inputs], 1)
                
                ids = ids[:, 1:].tolist()
                idss_predict.extend(ids)
                
                if data.step == data.total_step:
                    break
        
        return idss_predict, ratings

    def evaluate_final(self):
        idss_predicted, ratings = self.generate_predictions(self.test_data)
        tokens_test = [
            ids2tokens(ids[1:], self.corpus.word_dict) 
            for ids in self.test_data.seq.tolist()
        ]
        tokens_predict = [
            ids2tokens(ids, self.corpus.word_dict) 
            for ids in idss_predicted
        ]
        
        feature_batch = feature_detect(tokens_predict, self.corpus.feature_set)
        fcr = feature_coverage_ratio(feature_batch, self.corpus.feature_set)
        fmr = feature_matching_ratio(feature_batch, self.test_data.feature)
        
        print(f'{now_time()} Feature Coverage Ratio: {fcr:.4f}')
        print(f'{now_time()} Feature Matching Ratio: {fmr:.4f}')
        
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        results = {'exp': text_predict, 'score': ratings}
        df = pd.DataFrame(results)
        df.to_csv('trip_backbone_with_r/exp_results/final_predictions.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Att2Seq with Mutual Information Maximization')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--index_dir', type=str, required=True, help='Directory with data splits')
    parser.add_argument('--emsize', type=int, default=64, help='Embedding size')
    parser.add_argument('--nhid', type=int, default=512, help='Hidden size')
    parser.add_argument('--nlayers', type=int, default=2, help='LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--clip', type=float, default=5.0, help='Gradient clipping')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/', help='Model save directory')
    parser.add_argument('--vocab_size', type=int, default=20000, help='Vocabulary size')
    parser.add_argument('--endure_times', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--words', type=int, default=15, help='Words to generate')
    args = parser.parse_args()

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'{now_time()} Using device: {device}')

    orchestrator = TrainingOrchestrator(args, device)
    orchestrator.setup_directories()
    orchestrator.load_data()
    orchestrator.initialize_models()
    orchestrator.initialize_bert()
    orchestrator.setup_training_components()

    kl_weights = [0.1, 0.2,0.4]
    entropy_weights = [0.025, 0.02, 0.01, 0.005]
    
    orchestrator.run_training(kl_weights, entropy_weights)
    orchestrator.evaluate_final()


if __name__ == '__main__':
    main()