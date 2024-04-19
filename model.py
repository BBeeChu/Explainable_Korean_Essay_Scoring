import torch as th
from torch import nn
from torch.nn import Sigmoid, ReLU
from transformers import BertPreTrainedModel, BertConfig, BertModel, AutoModel
import torch.nn.functional as F

import numpy as np

import random



def weights_init(layer):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        

class POS_Encoder(nn.Module):
    def __init__(self, args, mean_score, num_sentence, warmup=False):
        super(POS_Encoder, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.sentence_lstm = nn.LSTM(558, self.hidden_dim, batch_first=True, bidirectional=True)
        self.doc_lstm = nn.LSTM(558, self.hidden_dim, batch_first=True, bidirectional=True)
        self.mlp = nn.Linear(self.hidden_dim*4, 1)
        self.num_sentence = num_sentence
        self.num_token = args.max_len
        self.dropout = nn.Dropout(args.dropout_rate)
        self.warmup = warmup
        
        self.sen_pos_w_omega = nn.Parameter(th.Tensor(
            self.hidden_dim*2, self.hidden_dim*2))
        self.sen_pos_b_omega = nn.Parameter(
            th.Tensor(1, self.hidden_dim*2))
        self.sen_pos_u_omega = nn.Parameter(
            th.Tensor(self.hidden_dim*2, 1))
        
        self.sen_w_omega = nn.Parameter(th.Tensor(
            200, 200))
        self.sen_b_omega = nn.Parameter(
            th.Tensor(1, 200))
        self.sen_u_omega = nn.Parameter(
            th.Tensor(200, 1))
        
        self.doc_w_omega = nn.Parameter(th.Tensor(
            self.hidden_dim*2, self.hidden_dim*2))
        self.doc_b_omega = nn.Parameter(
            th.Tensor(1, self.hidden_dim*2))
        self.doc_u_omega = nn.Parameter(
            th.Tensor(self.hidden_dim*2, 1))
        
        
        
        nn.init.kaiming_normal_(self.doc_w_omega)
        nn.init.kaiming_normal_(self.doc_u_omega)
        nn.init.kaiming_normal_(self.doc_b_omega)
        
        nn.init.kaiming_normal_(self.sen_w_omega)
        nn.init.kaiming_normal_(self.sen_u_omega)
        nn.init.kaiming_normal_(self.sen_b_omega)
        
        nn.init.kaiming_normal_(self.sen_pos_w_omega)
        nn.init.kaiming_normal_(self.sen_pos_u_omega)
        nn.init.kaiming_normal_(self.sen_pos_b_omega)
        
    def forward(self, sen_pos, doc_pos):
        sen_pos = F.one_hot(sen_pos.to(th.long), num_classes=558).to(th.float32)
        doc_pos = F.one_hot(doc_pos.to(th.long), num_classes=558).to(th.float32)
        batch, _, num_pos, _ = sen_pos.size()
        sen_pos = sen_pos.view(batch*self.num_sentence, num_pos, 558)
        

        sen_pos_embed, _ = self.sentence_lstm(sen_pos)
        
        sen_pos_attention_w = th.tanh(th.matmul(
            sen_pos_embed, self.sen_pos_w_omega) + self.sen_pos_b_omega) 
        # (batch_size*num_sen, num_pos, hidden_dim)
        sen_pos_attention_u = th.matmul(sen_pos_attention_w, self.sen_pos_u_omega)
        # (batch_size*num_sen, num_pos, 1)
        sen_pos_attention_score = F.softmax(sen_pos_attention_u, dim=1)
        
        sen_pos_attention_hidden = sen_pos_embed * sen_pos_attention_score
        # (batch_size*num_sen, num_pos, hidden_dim)
        
        pool_sen_pos = th.sum(sen_pos_attention_hidden, dim=1).squeeze()
        pool_sen_pos = pool_sen_pos.view(batch, self.num_sentence, -1)
        # (batch_size, num_sen, hidden_dim)
        
        sen_attention_w = th.tanh(th.matmul(
            pool_sen_pos, self.sen_w_omega) + self.sen_b_omega) 
        # (batch_size*num_sen, num_pos, hidden_dim)
        sen_attention_u = th.matmul(sen_attention_w, self.sen_u_omega)
        # (batch_size*num_sen, num_pos, 1)
        sen_attention_score = F.softmax(sen_attention_u, dim=1)
        
        sen_attention_hidden = pool_sen_pos * sen_attention_score
        # (batch_size*num_sen, num_pos, hidden_dim)
        
        pool_sen = th.sum(sen_attention_hidden, dim=1).squeeze()
        # (batch_size, hidden_dim)
        
        doc_pos_embed, _ = self.doc_lstm(doc_pos)
        
        doc_attention_w = th.tanh(th.matmul(
        doc_pos_embed, self.doc_w_omega) + self.doc_b_omega)
        # (batch_size, seq_len, hidden_dim)
        doc_attention_u = th.matmul(doc_attention_w, self.doc_u_omega)
        # (batch_size, seq_len, 1)
        doc_attention_score = F.softmax(doc_attention_u, dim=1)
        # doc_attention_score = F.softmax(doc_attention_u, dim=1)
        # (batch_size, seq_len, hidden_dim)
        doc_attention_hidden = doc_pos_embed * doc_attention_score
        # 加权求和 (batch_size, seq_len, hidden_dim)
        pool_doc_pos = th.sum(doc_attention_hidden, dim=1).squeeze()
        # (batch_size, hidden_dim)
        
        if self.warmup:    
            logit = self.dropout(self.mlp(th.cat([pool_sen, pool_doc_pos], dim=1)).squeeze() )
            return F.sigmoid(logit)
        else:
            return pool_sen_pos, doc_attention_hidden
            
            


class Saliency_Essay_Model(nn.Module):
    def __init__(self, args, mean_score, num_keyword, num_sentence, prompt_sen_embeddings, prompt_key_embeddings):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.pos_embedder = POS_Encoder(args, mean_score, num_sentence, warmup=False)
        self.num_keyword = num_keyword
        self.num_sentence = num_sentence
        self.num_token = args.max_len
        self.prompt_sen_embeddings = prompt_sen_embeddings
        self.prompt_key_embeddings = prompt_key_embeddings
        
        self.init_prompt_sen = nn.Linear(768, self.hidden_dim)

        self.init_sen = nn.LSTM(768, self.hidden_dim, batch_first=True)
        self.sen_w = nn.Parameter(th.Tensor(self.hidden_dim, self.hidden_dim))
        self.sen_b = nn.Parameter(th.Tensor(1, self.hidden_dim))
        self.sen_u = nn.Parameter(th.Tensor(self.hidden_dim, 1))
        self.init_key = nn.Linear(768, self.hidden_dim)
        # self.init_doc = nn.Linear(768, self.hidden_dim)
        
        self.sen_pos_sim = nn.Linear(self.num_sentence, self.hidden_dim)
        self.doc_pos_sim = nn.Linear(self.num_token, self.hidden_dim)
        
        self.sen_sim = nn.Linear(self.num_sentence, self.hidden_dim)
        self.key_sim = nn.Linear(self.num_keyword, self.hidden_dim)
        # self.doc_sim = nn.Linear(prompt_sen_embeddings.shape[1]*2+1, self.num_token)
        
        
        self.activation_function = nn.ReLU()
        # self.doc_mlp = nn.Sequential(
        #     nn.BatchNorm1d(self.num_token),
        #     nn.Linear(self.num_token, self.hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(args.dropout_rate),
        # )
        self.sen_mlp = nn.Sequential(
            nn.BatchNorm1d(self.num_sentence),
            nn.Linear(self.num_sentence, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout_rate),
        )
        self.key_mlp = nn.Sequential(
            nn.BatchNorm1d(self.num_keyword),
            nn.Linear(self.num_keyword, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout_rate),
        )
        self.doc_pos_mlp = nn.Sequential(
            nn.BatchNorm1d(self.num_token),
            nn.Linear(self.num_token, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout_rate),
        )
        self.sen_pos_mlp = nn.Sequential(
            nn.BatchNorm1d(self.num_sentence),
            nn.Linear(self.num_sentence, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout_rate),
        )
        
            
        
        self.exp_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim*4, 1)
        )
        
        self.org_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim*4, 1)
        )
        
        self.cont_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim*2, 1)
        )
        
        self.sen_pos_w_omega = nn.Parameter(th.Tensor(
            self.num_sentence, self.num_sentence))
        self.sen_pos_b_omega = nn.Parameter(
            th.Tensor(1, self.num_sentence))
        self.sen_pos_u_omega = nn.Parameter(
            th.Tensor(self.num_sentence, 1))
        
        self.doc_pos_w_omega = nn.Parameter(th.Tensor(
            self.num_token, self.num_token))
        self.doc_pos_b_omega = nn.Parameter(
            th.Tensor(1, self.num_token))
        self.doc_pos_u_omega = nn.Parameter(
            th.Tensor(self.num_token, 1))
        
        
        # self.doc_w_omega = nn.Parameter(th.Tensor(
        #     self.bert_model_config.hidden_size, self.bert_model_config.hidden_size))
        # self.doc_b_omega = nn.Parameter(
        #     th.Tensor(1, self.bert_model_config.hidden_size))
        # self.doc_u_omega = nn.Parameter(
        #     th.Tensor(self.bert_model_config.hidden_size, 1))
        
        self.sen_w_omega = nn.Parameter(th.Tensor(
            self.num_sentence, self.num_sentence))
        self.sen_b_omega = nn.Parameter(
            th.Tensor(1, self.num_sentence))
        self.sen_u_omega = nn.Parameter(
            th.Tensor(self.num_sentence, 1))
        
        self.key_w_omega = nn.Parameter(th.Tensor(
            self.num_keyword, self.num_keyword))
        self.key_b_omega = nn.Parameter(
            th.Tensor(1, self.num_keyword))
        self.key_u_omega = nn.Parameter(
            th.Tensor(self.num_keyword, 1))
        
        
        # nn.init.xavier_normal_(self.doc_w_omega)
        # nn.init.xavier_normal_(self.doc_u_omega)
        # nn.init.xavier_normal_(self.doc_b_omega)
        
        nn.init.xavier_normal_(self.doc_pos_w_omega)
        nn.init.xavier_normal_(self.doc_pos_u_omega)
        nn.init.xavier_normal_(self.doc_pos_b_omega)
        
        nn.init.xavier_normal_(self.sen_w_omega)
        nn.init.xavier_normal_(self.sen_u_omega)
        nn.init.xavier_normal_(self.sen_b_omega)
        
        nn.init.xavier_normal_(self.sen_pos_w_omega)
        nn.init.xavier_normal_(self.sen_pos_u_omega)
        nn.init.xavier_normal_(self.sen_pos_b_omega)
        
        nn.init.xavier_normal_(self.key_w_omega)
        nn.init.xavier_normal_(self.key_u_omega)
        nn.init.xavier_normal_(self.key_b_omega)
        
        
        
        self.num_ref = args.num_ref
        self.mean_score = mean_score
        
        
        
        # for layer in self.doc_mlp:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)
        #         layer.bias.data.fill_(self.mean_score)
        for layer in self.sen_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(self.mean_score)
        for layer in self.key_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(self.mean_score)
                
        for layer in self.exp_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(self.mean_score)
        for layer in self.org_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(self.mean_score)
        for layer in self.cont_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                layer.bias.data.fill_(self.mean_score)
        
        nn.init.xavier_normal_(self.key_sim.weight)
        self.key_sim.bias.data.fill_(self.mean_score)
        nn.init.xavier_normal_(self.sen_sim.weight)
        self.sen_sim.bias.data.fill_(self.mean_score)
        # nn.init.xavier_normal_(self.doc_sim.weight)
        # self.doc_sim.bias.data.fill_(self.mean_score)
        nn.init.xavier_normal_(self.sen_pos_sim.weight)
        self.sen_pos_sim.bias.data.fill_(self.mean_score)
        nn.init.xavier_normal_(self.doc_pos_sim.weight)
        self.doc_pos_sim.bias.data.fill_(self.mean_score)
        nn.init.xavier_normal_(self.init_prompt_sen.weight)
        self.init_prompt_sen.bias.data.fill_(self.mean_score)
        # nn.init.xavier_normal_(self.init_doc.weight)
        # self.init_doc.bias.data.fill_(self.mean_score)
        nn.init.xavier_normal_(self.init_key.weight)
        self.init_key.bias.data.fill_(self.mean_score)

    
    def cosine_similarity(self, v1, v2, dim=-1):
        
        cos_sim = th.matmul(v1, v2.transpose(dim-1,dim))/(th.norm(v1, dim=dim).unsqueeze(-1)*th.norm(v2, dim=dim).unsqueeze(-1))
        
        return cos_sim
    
    def saliency(self, embeddings_1, embeddings_2):
        with th.enable_grad():
           
            s1 = F.relu((embeddings_1*(embeddings_1.grad))).sum(dim=-1)
           
            s1 = self.minimax(s1)
           
            embeddings_1.grad = None
            embeddings_2.grad = None

            
        return s1.unsqueeze(-1)
        
        
    def minimax(self, v):
        v_min = v.min(dim=-1, keepdim=True)[0]
        v_max = v.max(dim=-1, keepdim=True)[0]
        
        return (v - v_min)/(v_max - v_min)

    def forward(self, prompt_ids, scoring_sen_embeddings, scoring_key_embeddings,
                scoring_sen_pos, ref_sen_pos, scoring_doc_pos, ref_doc_pos, test=False):

        scoring_sen_pos, scoring_doc_pos = self.pos_embedder(scoring_sen_pos, scoring_doc_pos)
        ref_sen_pos, ref_doc_pos = self.pos_embedder(ref_sen_pos, ref_doc_pos)
        

        
        
        pool_ref_sen_pos = th.mean(ref_sen_pos, dim=0).repeat(scoring_sen_pos.shape[0],1,1)
        # (batch_size, hidden_dim*2)
        
        pool_ref_doc_pos = th.mean(ref_doc_pos, dim=0).repeat(scoring_doc_pos.shape[0],1,1)
        # (batch_size, hidden_dim*2)
        
        
        with th.enable_grad():
            cop_scoring_sen_pos = scoring_sen_pos.clone().detach().requires_grad_(True)
            cop_scoring_doc_pos = scoring_doc_pos.clone().detach().requires_grad_(True)
            
            cop_pool_ref_sen_pos = pool_ref_sen_pos.clone().detach().requires_grad_(True)
            
            
            cop_pool_ref_doc_pos = pool_ref_doc_pos.clone().detach().requires_grad_(True)
            
            
            entire_sen_pos_sim = self.cosine_similarity(cop_scoring_sen_pos, cop_pool_ref_sen_pos)
            
            # (batch_size, )
            
            entire_doc_pos_sim = self.cosine_similarity(cop_scoring_doc_pos, cop_pool_ref_doc_pos)
            # (batch_size, )
            
            entire_sen_pos_sim.backward(th.full(entire_sen_pos_sim.shape, 1.0, device='cuda:0'))
            entire_doc_pos_sim.backward(th.full(entire_doc_pos_sim.shape, 1.0, device='cuda:0'))
            
            sen_pos_saliency = self.saliency(cop_scoring_sen_pos, cop_pool_ref_sen_pos)
            # (batch_size, num_sen, num_sen)
            doc_pos_saliency = self.saliency(cop_scoring_doc_pos, cop_pool_ref_doc_pos)
            # (batch_size, num_token, num_token)
        
        sen_pos_cos_sim = self.cosine_similarity(scoring_sen_pos, pool_ref_sen_pos)
        # (batch_size, num_sen, num_sen)
        
        doc_pos_cos_sim = self.cosine_similarity(scoring_doc_pos, pool_ref_doc_pos)
        # (batch_size, num_doc, num_doc)
        
        sen_pos_sal_sim = sen_pos_saliency*sen_pos_cos_sim
        # (batch_size, num_sen, num_sen)
        doc_pos_sal_sim = doc_pos_saliency*doc_pos_cos_sim
        # (batch_size, num_doc, num_doc)
        
        
        sen_pos_sim = self.sen_pos_mlp(sen_pos_sal_sim).mean(dim=1)
        # (batch_size, hidden_dim)
        doc_pos_sim = self.doc_pos_mlp(doc_pos_sal_sim).mean(dim=1)
        # (batch_size, hidden_dim)
        
        sen_prompt_embeddings = self.init_prompt_sen(self.prompt_sen_embeddings[th.LongTensor(prompt_ids)]).unsqueeze(1)
        
        # (batch_size, num_sen, 768)
        key_prompt_embeddings = self.init_key(self.prompt_key_embeddings[th.LongTensor(prompt_ids)])
        
        # (batch_size, num_key, 768)
        
        scoring_sen_embeddings, _ = self.init_sen(scoring_sen_embeddings)
        
        
        scoring_key_embeddings = self.init_key(scoring_key_embeddings)
        
        
        with th.enable_grad():
            
            cop_sen_prompt_embeddings = sen_prompt_embeddings.clone().detach().requires_grad_(True)
            cop_scoring_sen_embeddings = scoring_sen_embeddings.clone().detach().requires_grad_(True)
            
            entire_sen_sim = self.cosine_similarity(cop_scoring_sen_embeddings, cop_sen_prompt_embeddings)
            
            
            # (batch_size, )
            
            
            entire_sen_sim.backward(th.full(entire_sen_sim.shape, 1.0, device='cuda:0'))
            
            
            
            sen_saliency = self.saliency(cop_scoring_sen_embeddings,
                                        cop_sen_prompt_embeddings)
            
            
        # (batch_size, num_sen, num_sen)

        sen_cos_sim = self.cosine_similarity(scoring_sen_embeddings,
                                             sen_prompt_embeddings)
        # (batch_size, num_sen, num_sen)
        sal_sim_sen = sen_saliency*sen_cos_sim
        

        sim_sen = self.sen_mlp(sal_sim_sen.squeeze())
        # (batch_size, hidden_size)
        

        # (batch_size, num_sen)
        

        # (batch_size, hidden_dim)
                
        
        with th.enable_grad():
           
            cop_key_prompt_embeddings = key_prompt_embeddings.clone().detach().requires_grad_(True)
            cop_scoring_key_embeddings = scoring_key_embeddings.clone().detach().requires_grad_(True)
            
            entire_key_sim = self.cosine_similarity(cop_key_prompt_embeddings, cop_scoring_key_embeddings)
            
            # (batch_size,)
            entire_key_sim.backward(th.full(entire_key_sim.shape, 1.0, device='cuda:0'))
            
            key_saliency = self.saliency(cop_scoring_key_embeddings, cop_key_prompt_embeddings)
            # (batch_size, num_key, num_key)
        
        key_cos_sim = self.cosine_similarity(scoring_key_embeddings, key_prompt_embeddings)
        
        # (batch_size, num_key, num_key)
        sal_sim_key = key_saliency*key_cos_sim
        
        sim_key = self.key_mlp(sal_sim_key).mean(dim=1)
        
        
        exp_logit = self.exp_mlp(th.cat([sen_pos_sim, doc_pos_sim, sim_sen, sim_key], dim=-1)).squeeze(-1)
        org_logit = self.org_mlp(th.cat([sen_pos_sim, doc_pos_sim, sim_sen, sim_key], dim=-1)).squeeze(-1)
        cont_logit = self.cont_mlp(th.cat([sim_key, sim_sen], dim=-1)).squeeze(-1)
        
        pred_exp = F.sigmoid(exp_logit)
        pred_org = F.sigmoid(org_logit)
        pred_cont = F.sigmoid(cont_logit)
        pred_total = F.sigmoid(exp_logit+org_logit+cont_logit)

        if test:
            return (pred_exp, pred_org, pred_cont, pred_total), (sen_saliency, sen_cos_sim, sal_sim_sen), \
            (key_saliency, key_cos_sim, sal_sim_key), (sen_pos_saliency, sen_pos_cos_sim, sen_pos_sal_sim), (doc_pos_saliency, doc_pos_cos_sim, doc_pos_sal_sim)
        else:
            return (pred_exp, pred_org, pred_cont, pred_total)
    
