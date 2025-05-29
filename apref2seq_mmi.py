import os
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from module import APRef2Seq, Mine
from utils import (
    rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens,
    unique_sentence_percent, feature_detect, feature_matching_ratio,
    feature_coverage_ratio, feature_diversity
)
from transformers import AutoTokenizer, BertModel

class MutualInformationEstimator:
    def __init__(self, args, corpus, device):
        self.args = args
        self.corpus = corpus
        self.device = device
        self.setup_paths()
        self.load_models()
        self.setup_optimizers()
        self.load_aspect_resources()
        
    def setup_paths(self):
        if not os.path.exists(self.args.checkpoint):
            os.makedirs(self.args.checkpoint)
        self.model_path = os.path.join(self.args.checkpoint, 'model.pt')
        self.prediction_path = os.path.join(self.args.checkpoint, self.args.outf)
        
    def load_models(self):
        # Initialize models
        ntoken = len(self.corpus.word_dict)
        self.model = APRef2Seq(
            ntoken, self.args.nhid, self.args.encoder_dropout,
            self.args.decoder_dropout, self.args.nlayers, self.args.dmax
        ).to(self.device)
        
        self.text_criterion = nn.NLLLoss(ignore_index=self.corpus.word_dict.word2idx['<pad>'])
        
        # Load pretrained MINE
        self.mi_estimator = Mine(768 + 768, hidden_size=256).to(self.device)
        pretrain_mine_ckpt = torch.load('../MINE/checkpoints/tripadvisor_review_f_mi')
        self.mi_estimator.load_state_dict(pretrain_mine_ckpt['model_state_dict'])
        
        # Load reference model
        with open(self.model_path, 'rb') as f:
            self.ref_model = torch.load(f).to(self.device)
        
        # Initialize BERT encoder
        self.tokenizer = AutoTokenizer.from_pretrained("../bert_models/bert-base-cased")
        self.bert_encoder = BertModel.from_pretrained("../bert_models/my_tripadvisor_model").to(self.device)
        self.freeze_model(self.bert_encoder)
        
    def setup_optimizers(self):
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.002)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1)
        self.mine_net_optim = torch.optim.Adam(self.mi_estimator.parameters(), lr=1e-4)
        
    def load_aspect_resources(self):
        # Load aspect resources
        self.aspect_bert_token_id_dict = pickle.load(
            open('/data/run01/scv9803/zhaoyurou/dataset/tripadvisor/bert_tokenizer_feature_id.dict.pkl', 'rb')
        )
        self.aspect_bert_word_embds = np.load('/data/run01/scv9803/zhaoyurou/dataset/tripadvisor/bert_feature_word_embd.npy')
        self.topk = 50
        
    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def activate_model(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def mutual_information(self, joint, marginal, mine_net, batch=False):
        t = mine_net(joint)
        et = torch.exp(mine_net(marginal))
        if not batch:
            mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        else:
            mi_lb = t - torch.log(et)
        return mi_lb, t, et

    def train_text(self, data):
        self.model.train()
        text_loss = 0.
        total_sample = 0
        num_steps = int(data.sample_num / self.args.batch_size) + 1
        
        with tqdm(total=num_steps, desc='Training') as pbar:
            while True:
                user_refs, item_refs, aspect_ids, seq = data.next_batch()
                batch_size = user_refs.size(0)
                user_refs = user_refs.to(self.device)
                item_refs = item_refs.to(self.device)
                aspect_ids = aspect_ids.to(self.device)
                seq = seq.to(self.device)

                self.optimizer.zero_grad()
                log_word_prob = self.model(
                    user_refs, item_refs, aspect_ids, seq[:, :-1].transpose(0, 1)
                )
                loss = self.text_criterion(
                    log_word_prob.view(-1, len(self.corpus.word_dict)),
                    seq[:, 1:].transpose(0, 1).reshape((-1,))
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                text_loss += batch_size * loss.item()
                total_sample += batch_size
                pbar.update(1)
                
                if data.step == data.total_step:
                    break
                    
        return text_loss / total_sample

    def evaluate_text(self, data):
        self.model.eval()
        text_loss = 0.
        total_sample = 0
        num_steps = int(data.sample_num / self.args.batch_size) + 1
        
        with torch.no_grad(), tqdm(total=num_steps, desc='Evaluating') as pbar:
            while True:
                user_refs, item_refs, aspect_ids, seq = data.next_batch()
                batch_size = user_refs.size(0)
                user_refs = user_refs.to(self.device)
                item_refs = item_refs.to(self.device)
                aspect_ids = aspect_ids.to(self.device)
                seq = seq.to(self.device)

                log_word_prob = self.model(
                    user_refs, item_refs, aspect_ids, seq[:, :-1].transpose(0, 1)
                )
                loss = self.text_criterion(
                    log_word_prob.view(-1, len(self.corpus.word_dict)),
                    seq[:, 1:].transpose(0, 1).reshape((-1,))
                )

                text_loss += batch_size * loss.item()
                total_sample += batch_size
                pbar.update(1)
                
                if data.step == data.total_step:
                    break
                    
        return text_loss / total_sample

    def generate_text(self, data):
        self.model.eval()
        idss_predict = []
        aspects = []
        word2idx = self.corpus.word_dict.word2idx
        idx2word = self.corpus.word_dict.idx2word
        eos_idx = word2idx['<eos>']
        num_steps = int(data.sample_num / self.args.batch_size) + 1
        
        with torch.no_grad(), tqdm(total=num_steps, desc='Generating') as pbar:
            while True:
                user_refs, item_refs, aspect_ids, seq = data.next_batch()
                user_refs = user_refs.to(self.device)
                item_refs = item_refs.to(self.device)
                aspect_words = ids2tokens(aspect_ids, word2idx, idx2word)
                aspects.extend(aspect_words)
                aspect_ids = aspect_ids.to(self.device)
                
                inputs = seq[:, :1].to(self.device)
                decoder_hidden = None
                ids = inputs
                
                for idx in range(self.args.words):
                    if idx == 0:
                        outU, hiddenU = self.model.encoderU(user_refs)
                        outI, hiddenI = self.model.encoderI(item_refs)
                        aspect_embd = self.model.word_embeddings(aspect_ids).unsqueeze(0)
                        decoder_hidden = hiddenU[:self.model.decoder.n_layers] + hiddenI[:self.model.decoder.n_layers]
                        log_word_prob, decoder_hidden, _, _, _ = self.model.decoder(
                            inputs.transpose(0, 1), decoder_hidden, outU, outI, aspect_embd
                        )
                    else:
                        log_word_prob, decoder_hidden, _, _, _ = self.model.decoder(
                            inputs.transpose(0, 1), decoder_hidden, outU, outI, aspect_embd
                        )
                    
                    word_prob = log_word_prob.transpose(0, 1).squeeze().exp()
                    inputs = torch.argmax(word_prob, dim=1, keepdim=True)
                    ids = torch.cat([ids, inputs], 1)
                    
                    # Early stopping if all sequences have ended
                    if (inputs == eos_idx).all():
                        break
                
                idss_predict.extend(ids[:, 1:].tolist())
                pbar.update(1)
                
                if data.step == data.total_step:
                    break
                    
        return idss_predict, aspects

    def train_mi_estimator(self, data, val=False):
        self.mi_estimator.train()
        mi_results = []
        num_steps = int(data.sample_num / self.args.batch_size) + 1
        ma_et = 1.0
        ma_rate = 0.01
        
        with tqdm(total=num_steps, desc='MI Training') as pbar:
            while True:
                user_refs, item_refs, aspect_ids, seq = data.next_batch()
                batch_size = user_refs.size(0)
                aspect_words = ids2tokens(aspect_ids, self.corpus.word_dict.word2idx, self.corpus.word_dict.idx2word)
                
                # Prepare aspect embeddings
                aspect_bert_token_ids = [self.aspect_bert_token_id_dict[word] for word in aspect_words]
                aspect_word_embds = self.aspect_bert_word_embds[aspect_bert_token_ids]
                shuffled_aspect_word_embds = aspect_word_embds.copy()
                np.random.shuffle(shuffled_aspect_word_embds)
                aspect_word_embds = torch.tensor(aspect_word_embds)
                shuffled_aspect_word_embds = torch.tensor(shuffled_aspect_word_embds)
                
                # Generate text
                tokens_predict, _ = self.generate_batch_text(user_refs, item_refs, aspect_ids)
                text_predict = [' '.join(tokens) for tokens in tokens_predict]
                
                # Get BERT embeddings
                with torch.no_grad():
                    inputs = self.tokenizer(text_predict, return_tensors='pt', padding=True).to(self.device)
                    outputs = self.bert_encoder(**inputs)
                    encodings = outputs.pooler_output.cpu().numpy()
                
                embds = torch.tensor(encodings, dtype=torch.float32)
                joint = torch.cat([embds, aspect_word_embds], dim=1).float().to(self.device)
                marginal = torch.cat([embds, shuffled_aspect_word_embds], dim=1).float().to(self.device)
                
                # Compute MI
                mi_lb, t, marginal_t = self.mutual_information(joint, marginal, self.mi_estimator)
                mi_results.append(mi_lb.item())
                
                # Update MI estimator if not in validation mode
                if not val:
                    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(marginal_t)
                    loss = -(torch.mean(t) - (1 / ma_et) * torch.mean(marginal_t))
                    loss.backward()
                    self.mine_net_optim.step()
                    self.mine_net_optim.zero_grad()
                
                pbar.update(1)
                if data.step == data.total_step:
                    torch.cuda.empty_cache()
                    break
                    
        return mi_results

    def policy_gradient_train(self, data, val=False, **kwargs):
        status = 1
        self.model.train() if not val else self.model.eval()
        mc_sample_num = 5 if not val else 1
        
        # Initialize metrics
        metrics = {
            'text_loss': 0., 'total_sample': 0, 'total_scaled_pg_loss': 0,
            'total_mi_loss': 0, 'total_kl_loss': 0, 'total_entropy_loss': 0,
            'total_mi': 0., 'mi_rewards': [], 'kl_rewards': [], 'entropy_rewards': []
        }
        
        num_steps = int(data.sample_num / self.args.batch_size) + 1
        with tqdm(total=num_steps, desc='PG Training' if not val else 'PG Validation') as pbar:
            while True:
                batch_metrics = self.process_pg_batch(data, mc_sample_num, val, **kwargs)
                self.update_metrics(metrics, batch_metrics)
                
                pbar.update(1)
                if data.step == data.total_step:
                    torch.cuda.empty_cache()
                    break
                    
        # Compute final metrics
        final_metrics = {}
        for key in ['text_loss', 'total_scaled_pg_loss', 'total_mi_loss', 
                   'total_kl_loss', 'total_entropy_loss']:
            final_metrics[key] = metrics[key] / metrics['total_sample']
            
        for key in ['mi_rewards', 'kl_rewards', 'entropy_rewards']:
            final_metrics[key] = torch.mean(torch.tensor(metrics[key])).item() if metrics[key] else 0.0
            
        return final_metrics

    def process_pg_batch(self, data, mc_sample_num, val, **kwargs):
        user_refs, item_refs, aspect_ids, seq = data.next_batch()
        batch_size = user_refs.size(0)
        word2idx = self.corpus.word_dict.word2idx
        idx2word = self.corpus.word_dict.idx2word
        eos_idx = word2idx['<eos>']
        
        # Prepare aspect embeddings
        aspect_words = ids2tokens(aspect_ids, word2idx, idx2word)
        aspect_bert_token_ids = [self.aspect_bert_token_id_dict[word] for word in aspect_words]
        aspect_word_embds = self.aspect_bert_word_embds[aspect_bert_token_ids]
        shuffled_aspect_word_embds = aspect_word_embds.copy()
        np.random.shuffle(shuffled_aspect_word_embds)
        aspect_word_embds = torch.tensor(aspect_word_embds)
        shuffled_aspect_word_embds = torch.tensor(shuffled_aspect_word_embds)
        
      
        user_refs = user_refs.to(self.device)
        item_refs = item_refs.to(self.device)
        aspect_ids = aspect_ids.to(self.device)
        
        gen_log_prob_list = []
        ref_log_prob_list = []
        rewards = []
        mi_list = []
        
        # Monte Carlo sampling
        for i in range(mc_sample_num):
            inputs = seq[:, :1].to(self.device)
            ids = inputs
            probs = []
            ref_probs = []
            words_lens = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            # Generate sequence
            for idx in range(self.args.words):
                if idx == 0:
                    outU, hiddenU = self.model.encoderU(user_refs)
                    outI, hiddenI = self.model.encoderI(item_refs)
                    aspect_embd = self.model.word_embeddings(aspect_ids).unsqueeze(0)
                    decoder_hidden = hiddenU[:self.model.decoder.n_layers] + hiddenI[:self.model.decoder.n_layers]
                    
                    log_word_prob, decoder_hidden, _, _, _ = self.model.decoder(
                        inputs.transpose(0, 1), decoder_hidden, outU, outI, aspect_embd
                    )
                    
                    if kwargs.get('with_kl_control'):
                        with torch.no_grad():
                            ref_log_word_prob, _, _, _, _ = self.ref_model.decoder(
                                inputs.transpose(0, 1), decoder_hidden, outU, outI, aspect_embd
                            )
                else:
                    if kwargs.get('with_kl_control'):
                        with torch.no_grad():
                            ref_log_word_prob, _, _, _, _ = self.ref_model.decoder(
                                inputs.transpose(0, 1), decoder_hidden, outU, outI, aspect_embd
                            )
                    
                    log_word_prob, decoder_hidden, _, _, _ = self.model.decoder(
                        inputs.transpose(0, 1), decoder_hidden, outU, outI, aspect_embd
                    )
                
                # Sample next token
                word_prob = log_word_prob.transpose(0, 1).squeeze().exp()
                inputs = torch.multinomial(word_prob, 1)
                prob_var = word_prob.gather(-1, inputs).view(-1)
                probs.append(prob_var)
                
                if kwargs.get('with_kl_control'):
                    ref_prob_var = ref_log_word_prob.transpose(0, 1).squeeze().exp().gather(-1, inputs).view(-1)
                    ref_probs.append(ref_prob_var)
                
                # Update sequence lengths
                is_eos = (inputs == eos_idx).squeeze()
                not_end = words_lens == 0
                if idx != self.args.words - 1:
                    words_lens[not_end & is_eos] = idx + 1
                    if (words_lens != 0).all():
                        break
                else:
                    words_lens[not_end] = self.args.words
                
                ids = torch.cat([ids, inputs], 1)
            
            # Compute log probabilities
            probs = torch.stack(probs, dim=1)
            log_gen_probs = probs.log()
            for index, l in enumerate(words_lens):
                log_gen_probs[index, l:] = 0
            log_gen_probs = log_gen_probs.sum(-1)
            gen_log_prob_list.append(log_gen_probs)
            
            if kwargs.get('with_kl_control'):
                ref_probs = torch.stack(ref_probs, dim=1)
                ref_log_gen_probs = ref_probs.log()
                for index, l in enumerate(words_lens):
                    ref_log_gen_probs[index, l:] = 0
                ref_log_gen_probs = ref_log_gen_probs.sum(-1)
                ref_log_prob_list.append(ref_log_gen_probs)
            
            # Compute rewards
            tokens_predict = [ids2tokens(ids[:, 1:].tolist(), word2idx, idx2word)]
            text_predict = [' '.join(tokens) for tokens in tokens_predict]
            
            with torch.no_grad():
                inputs = self.tokenizer(text_predict, return_tensors='pt', padding=True).to(self.device)
                outputs = self.bert_encoder(**inputs)
                encodings = outputs.pooler_output.cpu().numpy()
            
            embds = torch.tensor(encodings, dtype=torch.float32)
            joint = torch.cat([embds, aspect_word_embds], dim=1).float().to(self.device)
            marginal = torch.cat([embds, shuffled_aspect_word_embds], dim=1).float().to(self.device)
            
            batch_mi, _, _ = self.mutual_information(joint, marginal, self.mi_estimator, batch=False)
            mi_list.append(batch_mi.item())
            mi, _, _ = self.mutual_information(joint, marginal, self.mi_estimator, batch=True)
            rewards.append(mi.squeeze())
        
        # Compute policy gradient loss
        gen_log_prob_list = torch.stack(gen_log_prob_list, dim=1)
        rewards = torch.stack(rewards, dim=1)
        
        if not val:
            mi_pg_loss = -((rewards - rewards.mean(-1, keepdim=True)) * gen_log_prob_list).mean()
        else:
            mi_pg_loss = -(rewards * gen_log_prob_list).mean().detach()
        
        # Additional loss components
        kl_pg_loss = 0
        entropy_pg_loss = 0
        if kwargs.get('with_kl_control'):
            ref_log_prob_list = torch.stack(ref_log_prob_list, dim=1)
            kl_reward = (gen_log_prob_list - ref_log_prob_list).detach()
            
            if not val:
                kl_pg_loss = ((kl_reward - kl_reward.mean(-1, keepdim=True)) * gen_log_prob_list).mean()
            else:
                kl_pg_loss = (kl_reward * gen_log_prob_list).mean().detach()
            
            if kwargs.get('add_entropy'):
                entropy_reward = (-gen_log_prob_list).detach()
                if not val:
                    entropy_pg_loss = -((entropy_reward - entropy_reward.mean(-1, keepdim=True)) * gen_log_prob_list).mean()
                else:
                    entropy_pg_loss = -(entropy_reward * gen_log_prob_list).mean().detach()
        
        # Combine losses
        scaled_pg_loss = (
            kwargs.get('mi_control_weight', 1.0) * mi_pg_loss +
            kwargs.get('kl_scale_weight', 1.0) * kwargs.get('kl_control_weight', 1.0) * kl_pg_loss +
            kwargs.get('entropy_scale_weight', 1.0) * kwargs.get('entropy_control_weight', 1.0) * entropy_pg_loss
        )
        
        # Update model if not in validation mode
        if not val:
            self.optimizer.zero_grad()
            scaled_pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
        
        # Prepare batch metrics
        batch_metrics = {
            'text_loss': batch_size * -(rewards * gen_log_prob_list).mean().item(),
            'scaled_pg_loss': batch_size * scaled_pg_loss.item(),
            'mi_loss': batch_size * mi_pg_loss.item(),
            'kl_loss': batch_size * kl_pg_loss.item(),
            'entropy_loss': batch_size * entropy_pg_loss.item(),
            'total_mi': batch_size * (sum(mi_list) / len(mi_list)),
            'batch_size': batch_size
        }
        
        return batch_metrics

    def update_metrics(self, metrics, batch_metrics):
        for key in ['text_loss', 'scaled_pg_loss', 'mi_loss', 'kl_loss', 'entropy_loss', 'total_mi']:
            metrics[key] += batch_metrics[key]
        metrics['total_sample'] += batch_metrics['batch_size']

    def compute_fmr(self, exps, aspects):
        match_count = 0
        for exp, aspect in zip(exps, aspects):
            if aspect in exp.split():
                match_count += 1
        return match_count / len(exps)

    def compute_avg_len(self, exps):
        return sum(len(exp.split()) for exp in exps) / len(exps)

    def run_training(self):
        # Initialize data loaders
        train_data = Batchify(self.corpus.train, self.corpus.word_dict.word2idx, 
                             self.args.words, self.args.batch_size, shuffle=True)
        val_data = Batchify(self.corpus.valid, self.corpus.word_dict.word2idx, 
                           self.args.words, self.args.batch_size)
        
        # Training state
        best_val_loss = float('inf')
        best_mi = float('-inf')
        best_fmr = 0.0
        endure_count = 0
        
        # Main training loop
        for epoch in range(1, self.args.epochs + 1):
            print(f'{now_time()} Epoch {epoch}/{self.args.epochs}')
            
            # Alternate between MI estimator and generator training
            if epoch % 2 == 0:
                print('Training MI estimator...')
                self.freeze_model(self.model)
                self.activate_model(self.mi_estimator)
                
                train_mi = self.train_mi_estimator(train_data)
                val_mi = self.train_mi_estimator(val_data, val=True)
                
                print(f'{now_time()} Train MI: {np.mean(train_mi):.4f}, Val MI: {np.mean(val_mi):.4f}')
                
                # Save MI estimator
                torch.save(self.mi_estimator.state_dict(), 
                          f'tripadvisor_efm_aspect_top{self.topk}_mmi/f_mi_estimator.pt')
            else:
                print('Training generator...')
                self.freeze_model(self.mi_estimator)
                self.activate_model(self.model)
                
                # Train with policy gradient
                train_metrics = self.policy_gradient_train(
                    train_data,
                    with_kl_control=True,
                    add_entropy=True,
                    kl_control_weight=0.8,
                    entropy_control_weight=0.01,
                    kl_scale_weight=0.2,
                    entropy_scale_weight=0.05,
                    tau=0.1
                )
                
                # Validate
                val_metrics = self.policy_gradient_train(
                    val_data,
                    val=True,
                    with_kl_control=True,
                    add_entropy=True,
                    kl_control_weight=0.8,
                    entropy_control_weight=0.01,
                    kl_scale_weight=0.2,
                    entropy_scale_weight=0.05,
                    tau=0.1
                )
                
                # Generate and compute metrics
                idss_predicted, aspects = self.generate_text(val_data)
                tokens_predict = [ids2tokens(ids, self.corpus.word_dict.word2idx, 
                                            self.corpus.word_dict.idx2word) for ids in idss_predicted]
                text_predict = [' '.join(tokens) for tokens in tokens_predict]
                val_fmr = self.compute_fmr(text_predict, aspects)
                val_exp_len = self.compute_avg_len(text_predict)
                
                print(f'{now_time()} Val FMR: {val_fmr:.4f}, Avg Len: {val_exp_len:.2f}')
                
                # Check for improvement
                if val_metrics['scaled_pg_loss'] < best_val_loss or val_metrics['mi_rewards'] > best_mi or val_fmr > best_fmr:
                    endure_count = 0
                    if val_metrics['scaled_pg_loss'] < best_val_loss:
                        best_val_loss = val_metrics['scaled_pg_loss']
                    if val_metrics['mi_rewards'] > best_mi:
                        best_mi = val_metrics['mi_rewards']
                    if val_fmr > best_fmr:
                        best_fmr = val_fmr
                        # Save best model
                        torch.save(self.model, 
                                  f'tripadvisor_efm_aspect_top{self.topk}_mmi/best_model_fmr{best_fmr:.4f}.pt')
                else:
                    endure_count += 1
                    if endure_count >= self.args.endure_times:
                        print(f'{now_time()} Early stopping at epoch {epoch}')
                        break
        
        print(f'Training completed. Best FMR: {best_fmr:.4f}')

    def evaluate_final(self):
        # Load best model
        self.model = torch.load(f'tripadvisor_efm_aspect_top{self.topk}_mmi/best_model_*.pt')
        

        test_data = Batchify(self.corpus.test, self.corpus.word_dict.word2idx, 
                            self.args.words, self.args.batch_size)
        

        test_loss = self.evaluate_text(test_data)
        print(f'{now_time()} Test PPL: {math.exp(test_loss):.4f}')

        idss_predicted, aspects = self.generate_text(test_data)
        tokens_predict = [ids2tokens(ids, self.corpus.word_dict.word2idx, 
                                    self.corpus.word_dict.idx2word) for ids in idss_predicted]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        
   
        results = pd.DataFrame({'exp': text_predict, 'aspect': aspects})
        results.to_csv(os.path.join(self.args.checkpoint, 'test_results.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='AP-Ref2seq with MMI Training')
    parser.add_argument('--data_path', type=str, required=True, help='Path for loading the pickle data')
    parser.add_argument('--ref_path', type=str, required=True, help='Path for loading the user/item references')
    parser.add_argument('--aspect_path', type=str, required=True, help='Path for loading the planned aspects')
    parser.add_argument('--index_dir', type=str, required=True, help='Load indexes')
    parser.add_argument('--nhid', type=int, default=256, help='Number of hidden units and size of word embedding')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of encoder/decoder layers')
    parser.add_argument('--encoder_dropout', type=float, default=0.5, help='Encoder dropout rate')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Decoder dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--clip', type=float, default=5.0, help='Gradient clipping')
    parser.add_argument('--epochs', type=int, default=100, help='Upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/', help='Directory to save the final model')
    parser.add_argument('--outf', type=str, default='generated.txt', help='Output file for generated text')
    parser.add_argument('--vocab_size', type=int, default=20000, help='Keep the most frequent words in the vocabulary')
    parser.add_argument('--endure_times', type=int, default=5, help='Maximum endure times of loss increasing on validation')
    parser.add_argument('--words', type=int, default=15, help='Number of words to generate for each sample')
    parser.add_argument('--dmax', type=int, default=5, help='Max number of documents')
    args = parser.parse_args()


    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    print(f'{now_time()} Loading data')
    corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size, args.ref_path, args.aspect_path)
    

    trainer = MutualInformationEstimator(args, corpus, device)
    

    trainer.run_training()
    
 
    trainer.evaluate_final()


if __name__ == '__main__':
    main()