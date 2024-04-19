import os
import random
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset
from loss import class_wise_mae_subscores
from tqdm import tqdm
import json
import re
import emoji
import ast
import pickle
from soynlp.normalizer import repeat_normalize
from torch.optim import SGD, Adam, AdamW
import torch.optim as optim


emojis = list({y for x in emoji.UNICODE_EMOJI.values() for y in x.keys()})
emojis = ''.join(emojis)
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

def load_base_config(path='./configs.json'):
    with open(path) as f:
        config = json.load(f)
        print('Base configs loaded.')
    return config




def cross_prompt_load_data(data_path, args, train_prompt, test_prompt):
    """_summary_

    Args:
        data_path (str): path of dataset
        args.school_level (str): level of school (e.g. elementary, middle, high)
        args.measure (str): types of measurement (e.g. total_exp, total_org, total_cont)
        args.bert_model (str): types of bert model
        args.max_len (int): maximal length of inputed essay
        
    return:
        input_ids (int): the token ids of raw essays
        token_type_ids (int): the type ids of input_ids
        attention_masks (int): attention masks of input_ids
        position_ids (int): position encoding of input_ids
        scores (float): groun truth scores

    """    
    
    # data loading section
    
    data = pd.read_csv(data_path)
    
    sen_pos = np.array(pd.read_pickle('./data/pos/sentence_pos.pkl'))
    doc_pos = np.array(pd.read_pickle('./data/pos/essay_pos.pkl'))
    sentence_embeddings = np.array(pd.read_pickle('./data/sentence_embeddings.pkl'))
    sentences = np.array(pd.read_pickle('./data/sentences.pkl'))
    keyword_embeddings = np.array(pd.read_pickle('./data/keyword_embeddings.pkl'))
    keywords = np.array(pd.read_pickle('./data/keywords.pkl'))
    
    
    total_prompt = train_prompt + test_prompt
    
    
    # label scaling (0.0-1.0)
    for p in total_prompt:
        tmp_data = data[data['ESSAY_SUBJECT'] == p]
        raw_exp_scores = tmp_data['total_exp']
        raw_org_scores = tmp_data['total_org']
        raw_cont_scores = tmp_data['total_cont']
        raw_total_scores = tmp_data['total_score']
    
        exp_min_score = raw_exp_scores.min()
        exp_max_score = raw_exp_scores.max()
        scaled_exp_scores = ((raw_exp_scores-exp_min_score)*(10-0))/(exp_max_score-exp_min_score)
        org_min_score = raw_org_scores.min()
        org_max_score = raw_org_scores.max()
        scaled_org_scores = ((raw_org_scores-org_min_score)*(10-0))/(org_max_score-org_min_score)
        cont_min_score = raw_cont_scores.min()
        cont_max_score = raw_cont_scores.max()
        scaled_cont_scores = ((raw_cont_scores-cont_min_score)*(10-0))/(cont_max_score-cont_min_score)
        total_min_score = raw_total_scores.min()
        total_max_score = raw_total_scores.max()
        scaled_total_scores = ((raw_total_scores-total_min_score)*(10-0))/(total_max_score-total_min_score)
    
        data.loc[data['ESSAY_SUBJECT']==p, 'total_exp'] = scaled_exp_scores/10
        data.loc[data['ESSAY_SUBJECT']==p, 'total_org'] = scaled_org_scores/10
        data.loc[data['ESSAY_SUBJECT']==p, 'total_cont'] = scaled_cont_scores/10
        data.loc[data['ESSAY_SUBJECT']==p, 'total_score'] = scaled_total_scores/10
        
    scaled_total_scores = data['total_score']
    
    # extracting top k reference essay and train data
    
    train_data = data[data['ESSAY_SUBJECT'].isin(train_prompt)][["ESSAY_SUBJECT", "ESSAY_CONTENT", args.measure]]
    top_k_indices = train_data.nlargest(args.num_ref, columns=args.measure).index.tolist()
    reference_data = train_data.loc[top_k_indices].values
    train_data = train_data.drop(top_k_indices)
    train_indices = train_data.index
    train_data = train_data.values
    
    # extracting mean score for bias init
    mean_score = scaled_total_scores[train_indices].mean().astype(float)
    
    # extracting test data
    test_data = data[data['ESSAY_SUBJECT'].isin(test_prompt)][["ESSAY_SUBJECT", "ESSAY_CONTENT", args.measure]]
    test_index_list = list()
    for i in range(6):
        test_index_list.append(test_data[test_data['ESSAY_SUBJECT']==test_prompt[i]].index)
    
    test_data = test_data.values
    
    
    # preprocessing reference essays
    reference_raw_essay = reference_data[:,0]
    reference_sentences = sentences[top_k_indices]
    reference_keywords = keywords[top_k_indices]
    reference_sen_pos = sen_pos[top_k_indices]
    reference_doc_pos = doc_pos[top_k_indices]
    
    
    
    total_data = data.values
    total_subjects = total_data[:,0]
    total_essay = total_data[:,1]
    total_scores = total_data[:,2:].astype(float)
    prompt_ids = pd.read_pickle('./data/prompt/prompt_ids.pkl')
    prompt_sen_embeddings = th.tensor(pd.read_pickle('./data/prompt/prompt_sentence_embeddings.pkl'))
    prompt_key_embeddings = th.tensor(pd.read_pickle('./data/prompt/prompt_keyword_embeddings.pkl'))
    prompts = list()
    for s in total_subjects:
        prompts.append(prompt_ids[s])
    prompt_ids = th.tensor(prompts)
    
    
    ids = []
    for i,e in enumerate(tqdm(total_essay)):
        ids.append(i)
    
    reference_ids = []
    
    for i,e in enumerate(tqdm(reference_raw_essay)):
        reference_ids.append(i)
    
    
    
    preprocessed_data = (th.LongTensor(ids), th.FloatTensor(sentence_embeddings), th.FloatTensor(keyword_embeddings), 
                         th.FloatTensor(total_scores), sentences, keywords, prompt_ids, th.LongTensor(sen_pos), th.LongTensor(doc_pos))
    
    references = (th.LongTensor(reference_ids), th.LongTensor(reference_sen_pos), th.LongTensor(reference_doc_pos))
    
    return preprocessed_data, references, mean_score, train_indices, test_index_list, prompt_sen_embeddings, prompt_key_embeddings

def dummy_cross_prompt_load_data(data_path, args, train_prompt, test_prompt):
    """_summary_

    Args:
        data_path (str): path of dataset
        args.school_level (str): level of school (e.g. elementary, middle, high)
        args.measure (str): types of measurement (e.g. total_exp, total_org, total_cont)
        args.bert_model (str): types of bert model
        args.max_len (int): maximal length of inputed essay
        
    return:
        input_ids (int): the token ids of raw essays
        token_type_ids (int): the type ids of input_ids
        attention_masks (int): attention masks of input_ids
        position_ids (int): position encoding of input_ids
        scores (float): groun truth scores

    """    
    
    data = pd.read_csv(data_path)[['ESSAY_SUBJECT', 'ESSAY_CONTENT', 'total_exp', 'total_org', 'total_cont', 'total_score']]
    # aug_data = pd.read_csv('./data/aug_data.csv')[['ESSAY_SUBJECT', 'ESSAY_CONTENT', 'total_exp', 'total_org', 'total_cont', 'total_score']]
    aug_data = pd.read_csv('./data/new_aug_data.csv')[['ESSAY_SUBJECT', 'ESSAY_CONTENT', 'total_exp', 'total_org', 'total_cont', 'total_score']]
    
    data = pd.concat([data, aug_data]).reset_index(drop=True)
    sen_pos_ids = np.array(pd.read_pickle('./data/pos/new_aug_new_sentence_pos.pkl'))
    doc_pos_ids = np.array(pd.read_pickle('./data/pos/new_aug_new_essay_pos.pkl'))
    dummy_sen_pos_ids = np.array(pd.read_pickle('./data/pos/dummy_sentence_pos.pkl'))
    dummy_doc_pos_ids = np.array(pd.read_pickle('./data/pos/dummy_essay_pos.pkl'))
    entire_sentence_embeddings = np.array(pd.read_pickle('./data/new_aug_{}_sentence_embeddings.pkl'.format(args.num_sentence)))
    entire_sentences = np.array(pd.read_pickle('./data/new_aug_{}_sentences.pkl'.format(args.num_sentence)))
    entire_keyword_embeddings = np.array(pd.read_pickle('./data/new_aug_{}_keyword_embeddings.pkl'.format(args.num_keyword)))
    entire_keywords = np.array(pd.read_pickle('./data/new_aug_{}_keywords.pkl'.format(args.num_keyword)))
    dummy_sentence_embeddings = np.array(pd.read_pickle('./data/dummy_{}_sentence_embeddings.pkl'.format(args.num_sentence)))
    dummy_sentences = np.array(pd.read_pickle('./data/dummy_{}_sentences.pkl'.format(args.num_sentence)))
    dummy_keyword_embeddings = np.array(pd.read_pickle('./data/dummy_{}_keyword_embeddings.pkl'.format(args.num_keyword)))
    dummy_keywords = np.array(pd.read_pickle('./data/dummy_{}_keywords.pkl'.format(args.num_keyword)))
    
    total_prompt = train_prompt + test_prompt
    
    
    for p in total_prompt:
        tmp_data = data[data['ESSAY_SUBJECT'] == p]
        raw_exp_scores = tmp_data['total_exp']
        raw_org_scores = tmp_data['total_org']
        raw_cont_scores = tmp_data['total_cont']
        raw_total_scores = tmp_data['total_score']
    
        exp_min_score = raw_exp_scores.min()
        exp_max_score = raw_exp_scores.max()
        scaled_exp_scores = ((raw_exp_scores-exp_min_score)*(10-0))/(exp_max_score-exp_min_score)
        org_min_score = raw_org_scores.min()
        org_max_score = raw_org_scores.max()
        scaled_org_scores = ((raw_org_scores-org_min_score)*(10-0))/(org_max_score-org_min_score)
        cont_min_score = raw_cont_scores.min()
        cont_max_score = raw_cont_scores.max()
        scaled_cont_scores = ((raw_cont_scores-cont_min_score)*(10-0))/(cont_max_score-cont_min_score)
        total_min_score = raw_total_scores.min()
        total_max_score = raw_total_scores.max()
        scaled_total_scores = ((raw_total_scores-total_min_score)*(10-0))/(total_max_score-total_min_score)
    
        data.loc[data['ESSAY_SUBJECT']==p, 'total_exp'] = scaled_exp_scores/10
        data.loc[data['ESSAY_SUBJECT']==p, 'total_org'] = scaled_org_scores/10
        data.loc[data['ESSAY_SUBJECT']==p, 'total_cont'] = scaled_cont_scores/10
        data.loc[data['ESSAY_SUBJECT']==p, 'total_score'] = scaled_total_scores/10
        
    scaled_total_scores = data['total_score']
    
    prompt = '미래 도시에 대한 본인의 생각'
    
    # filtered_data = data[(data['ESSAY_LEN']<=500)&(data['ESSAY_SUBJECT']==args.prompt)].reset_index(drop=True)
    
    train_data = data[data['ESSAY_SUBJECT'].isin(train_prompt)]
    train_data = train_data[["ESSAY_SUBJECT", "ESSAY_CONTENT", args.measure]]
    # filtered_aug_data = aug_data[['ESSAY_SUBJECT', 'ESSAY_CONTENT', args.measure]]
    # train_data = pd.concat([train_data, filtered_aug_data])
    top_k_indices = train_data.nlargest(args.num_ref, columns=args.measure).index.tolist()
    reference_data = train_data.loc[top_k_indices].values
    train_data = train_data.drop(top_k_indices)
    train_indices = train_data.index
    train_data = train_data.values
    
    mean_score = scaled_total_scores[train_indices].mean().astype(float)
    
    test_data = pd.read_pickle("./data/dummpy_data.pkl")
    
    
    test_data = np.array(test_data)
    
    
    # preprocessing reference essays
    reference_raw_essay = reference_data[:,0]
    reference_sentences = entire_sentences[top_k_indices]
    reference_keywords = entire_keywords[top_k_indices]
    reference_sen_pos = sen_pos_ids[top_k_indices]
    reference_doc_pos = doc_pos_ids[top_k_indices]
    # reference_scores = reference_data[:,5].astype(float).round()
    
    # with open('./result/cross_prompt/reference_essays.pkl', 'wb') as f:
    #     pickle.dump(reference_raw_essay, f)
    # with open('./result/cross_prompt/reference_sentences.pkl', 'wb') as f:
    #     pickle.dump(reference_sentences, f)
    # with open('./result/cross_prompt/reference_keywords.pkl', 'wb') as f:
    #     pickle.dump(reference_keywords, f)
    
    # filtered_data = data.dropna(
    #     subset=["ESSAY_SUBJECT", "ESSAY_LEN", "ESSAY_CONTENT", args.measure])[["ESSAY_SUBJECT", 
    #                                                              "ESSAY_LEN", "ESSAY_CONTENT", 'ttr', 'scaled_sentence_mean_length', 
    #                                                              'scaled_essay_length', 'total_exp', 'total_org', 'total_cont', 'total_score']].values
    
    # filtered_data = data.values
    # filtered_aug_data = aug_data.values
    # total_subjects = filtered_data[:,0]
    # total_essay = filtered_data[:,1]
    # total_scores = filtered_data[:,2:].astype(float)
    prompt_ids = pd.read_pickle('./data/preprocessed_data/prompt_ids.pkl')
    # prompt_sen_embeddings = th.tensor(pd.read_pickle('./data/preprocessed_data/prompt_sentence_embeddings.pkl'))
    prompt_sen_embeddings = th.tensor(pd.read_pickle('./data/preprocessed_data/prompt_whole_sen_embeddings.pkl'))
    prompt_key_embeddings = th.tensor(pd.read_pickle('./data/preprocessed_data/prompt_keyword_embeddings.pkl'))
    prompts = list()
    for i in range(len(test_data)):
        prompts.append(prompt_ids[prompt])
    prompt_ids = th.tensor(prompts)
    
    
    # tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    
    # ids = []
    # input_ids = []
    # attention_masks = []
    # for i,e in enumerate(tqdm(test_data)):
    #     ids.append(i)
    #     processed_essay = "[CLS] "+clean(e.replace("<span>#@문장구분#</span>","").replace("\n",""))
    #     encoded_essay = tokenizer.encode_plus(
    #         processed_essay,
    #         return_token_type_ids = True,
    #         return_attention_mask = True,
    #         return_tensors = "pt",
    #         max_length = args.max_len,
    #         padding='max_length',
    #         truncation=True
    #     )
        
    #     input_ids.append(encoded_essay['input_ids'])            
    #     attention_masks.append(encoded_essay['attention_mask'])
    
  
    
    # Creating position IDs
    
    reference_ids = []
    
    for i,e in enumerate(tqdm(reference_raw_essay)):
        reference_ids.append(i)
    
    
    
    preprocessed_data = (th.LongTensor(ids), th.LongTensor(th.cat(input_ids, dim=0)), th.LongTensor(th.cat(attention_masks, dim=0)), th.FloatTensor(entire_sentence_embeddings), \
        th.FloatTensor(entire_keyword_embeddings), entire_sentences, entire_keywords, prompt_ids, th.LongTensor(dummy_sen_pos_ids), th.LongTensor(dummy_doc_pos_ids), th.FloatTensor(dummy_sentence_embeddings),
        th.FloatTensor(dummy_keyword_embeddings), dummy_sentences, dummy_keywords)
    
    references = (th.LongTensor(reference_ids), reference_sentences, reference_keywords, th.LongTensor(reference_sen_pos), th.LongTensor(reference_doc_pos))
    return preprocessed_data, references, prompt_sen_embeddings, prompt_key_embeddings, mean_score



class MyDataset(Dataset):
    def __init__(self, dic):
        self.ids = dic["ids"]
        self.sentence_embeddings = dic["sentence_embeddings"]
        self.keyword_embeddings = dic["keyword_embeddings"]
        self.prompt_ids = dic["prompt_ids"]
        self.sen_pos = dic["sen_pos"]
        self.doc_pos = dic["doc_pos"]
        self.scores = dic["scores"]
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ids = self.ids[index]
        sentence_embeddings = self.sentence_embeddings[index]
        keyword_embeddings = self.keyword_embeddings[index]
        prompt_ids = self.prompt_ids[index]
        sen_pos = self.sen_pos[index]
        doc_pos = self.doc_pos[index]
        score = self.scores[index]
        
        return (ids, sentence_embeddings, keyword_embeddings, prompt_ids, sen_pos, doc_pos), score
    
class Dummy_MyDataset(Dataset):
    def __init__(self, ids, input_ids, attention_masks, sentence_embeddings, keyword_embeddings, 
                prompt_ids, sen_pos, doc_pos, dummy_sentence_embeddings, dummy_keyword_embeddings):
        self.ids = ids
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.sentence_embeddings = sentence_embeddings
        self.keyword_embeddings = keyword_embeddings
        self.prompt_ids = prompt_ids
        self.sen_pos = sen_pos
        self.doc_pos = doc_pos
        self.dummy_sentence_embeddings = dummy_sentence_embeddings
        self.dummy_keyword_embeddings = dummy_keyword_embeddings
        

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        ids = self.ids[index]
        input_ids = self.input_ids[index]
        attention_masks = self.attention_masks[index]
        sentence_embeddings = self.sentence_embeddings[index]
        keyword_embeddings = self.keyword_embeddings[index]
        prompt_ids = self.prompt_ids[index]
        sen_pos = self.sen_pos[index]
        doc_pos = self.doc_pos[index]
        dummy_sentence_embeddings = self.dummy_sentence_embeddings[index]
        dummy_keyword_embeddings = self.dummy_keyword_embeddings[index]

        
        return (ids, input_ids, attention_masks, sentence_embeddings, keyword_embeddings,
                prompt_ids, sen_pos, doc_pos, dummy_sentence_embeddings, dummy_sentence_embeddings)
    
    
class EarlyStopping(object):
    """score로 stopping하기"""

    def __init__(self, patience, save_path, eps):
        self.max_loss = 100000000
        self.patience = patience
        self.path = save_path
        self.eps = eps
        self.counter = 0

    def should_stop(self, model, previous_loss, loss):
        if loss <= self.max_loss:
            self.max_loss = loss
            self.counter = 0
            th.save({"model_state_dict": model.state_dict()}, os.path.join(
                self.path, "best_model_epoch.ckpt"))
            print("the best model has been saved by early stopping")
        if loss < previous_loss:
            self.counter = 0
        elif loss >= previous_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
    
def model_train(train_data, valid_data, model, references, ckpt_path, args):
    
    if th.cuda.is_available() and args.gpu != -1:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"
        
    if args.optimizer == "sgd":
        opt = SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        opt = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        opt = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.1)
    
    
    early_stopper = EarlyStopping(patience=args.patience, save_path=ckpt_path, eps=0.0001)
    
    mse_fuction = nn.MSELoss()
    mae_function = nn.L1Loss()
       
    
    previous_score = 0.0
    print("Begin Training...")
    print("Initial Learning Rate:", opt.param_groups[0]['lr'])
    
    for epoch in range(1, args.num_epochs+1):
        train_exp_mse_mean = 0.0
        train_org_mse_mean = 0.0
        train_cont_mse_mean = 0.0
        train_total_mse_mean = 0.0
        train_exp_mae_mean = 0.0
        train_org_mae_mean = 0.0
        train_cont_mae_mean = 0.0
        train_total_mae_mean = 0.0
        valid_exp_mse_mean = 0.0
        valid_org_mse_mean = 0.0
        valid_cont_mse_mean = 0.0
        valid_total_mse_mean = 0.0
        valid_exp_mae_mean = 0.0
        valid_org_mae_mean = 0.0
        valid_cont_mae_mean = 0.0
        valid_total_mae_mean = 0.0
        
        
        for i, train in enumerate(tqdm(train_data)):
            essay_data, scores = train
            id = essay_data[0]
            scoring_sen_embeddings = essay_data[1].to(device)
            scoring_key_embeddings = essay_data[2].to(device)
            prompt_ids = essay_data[3]
            scoring_sen_pos = essay_data[4].to(device)
            scoring_doc_pos = essay_data[5].to(device)
            ref_id = references[0]
            ref_sen_pos = references[1].to(device)
            ref_doc_pos = references[2].to(device)
            true_scores = scores.to(device)
            
            
            model.train()
            
            opt.zero_grad()
            
            pred_score = model(prompt_ids, scoring_sen_embeddings, scoring_key_embeddings, scoring_sen_pos, ref_sen_pos, scoring_doc_pos, ref_doc_pos)
            
            
            exp_mse = mse_fuction(pred_score[0], true_scores[:,0])
            org_mse = mse_fuction(pred_score[1], true_scores[:,1])
            cont_mse = mse_fuction(pred_score[2], true_scores[:,2])
            total_mse = mse_fuction(pred_score[3], true_scores[:,3])
            
            exp_mae = mae_function(pred_score[0], true_scores[:,0])
            org_mae = mae_function(pred_score[1], true_scores[:,1])
            cont_mae = mae_function(pred_score[2], true_scores[:,2])
            total_mae = mae_function(pred_score[3], true_scores[:,3])
            
            
            train_exp_mse_mean += exp_mse.data
            train_org_mse_mean += org_mse.data
            train_cont_mse_mean += cont_mse.data
            train_total_mse_mean += total_mse.data
            train_exp_mae_mean += exp_mae.data
            train_org_mae_mean += org_mae.data
            train_cont_mae_mean += cont_mae.data
            train_total_mae_mean += total_mae.data
            
            combined_mae = (exp_mae + org_mae + cont_mae + total_mae)/4
            combined_mse = (exp_mse + org_mse + cont_mse + total_mse)/4
            mae_mse = (combined_mae + combined_mse)/2
            
            if args.loss == "both":
                mae_mse.backward()
            elif args.loss == "mse":
                combined_mse.backward()
            elif args.loss == 'mae':
                combined_mae.backward()
            
            th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            
            opt.step()
            
        scheduler.step()
        print(f"Learning Rate of Epoch {epoch+1}:", opt.param_groups[0]['lr'])
        
        
        train_exp_mse_mean = train_exp_mse_mean/i
        train_org_mse_mean = train_org_mse_mean/i
        train_cont_mse_mean = train_cont_mse_mean/i
        train_total_mse_mean = train_total_mse_mean/i
        train_exp_mae_mean = train_exp_mae_mean/i
        train_org_mae_mean = train_org_mae_mean/i
        train_cont_mae_mean = train_cont_mae_mean/i
        train_total_mae_mean = train_total_mae_mean/i
        
        with th.no_grad():
            valid_exp_predictions_list = list()
            valid_org_predictions_list = list()
            valid_cont_predictions_list = list()
            valid_total_predictions_list = list()
            valid_exp_scores_list = list()
            valid_org_scores_list = list()
            valid_cont_scores_list = list()
            valid_total_scores_list = list()
            
            
            for ii, valid in enumerate(tqdm(valid_data)):
                essay_data, scores = valid
                id = essay_data[0]
                scoring_sen_embeddings = essay_data[1].to(device)
                scoring_key_embeddings = essay_data[2].to(device)
                prompt_ids = essay_data[3]
                scoring_sen_pos = essay_data[4].to(device)
                scoring_doc_pos = essay_data[5].to(device)
                ref_id = references[0]
                ref_sen_pos = references[1].to(device)
                ref_doc_pos = references[2].to(device)
                true_scores = scores.to(device)
                
                
                model.eval()

                
                pred_score = model(prompt_ids, scoring_sen_embeddings, scoring_key_embeddings, scoring_sen_pos, ref_sen_pos, scoring_doc_pos, ref_doc_pos)
            
                exp_mse = mse_fuction(pred_score[0], true_scores[:,0])
                org_mse = mse_fuction(pred_score[1], true_scores[:,1])
                cont_mse = mse_fuction(pred_score[2], true_scores[:,2])
                total_mse = mse_fuction(pred_score[3], true_scores[:,3])
                
                exp_mae = mae_function(pred_score[0], true_scores[:,0])
                org_mae = mae_function(pred_score[1], true_scores[:,1])
                cont_mae = mae_function(pred_score[2], true_scores[:,2])
                total_mae = mae_function(pred_score[3], true_scores[:,3])
                
                
                
                valid_exp_mse_mean += exp_mse.data
                valid_org_mse_mean += org_mse.data
                valid_cont_mse_mean += cont_mse.data
                valid_total_mse_mean += total_mse.data
                valid_exp_mae_mean += exp_mae.data
                valid_org_mae_mean += org_mae.data
                valid_cont_mae_mean += cont_mae.data
                valid_total_mae_mean += total_mae.data
                
                valid_exp_predictions_list.extend(
                    np.array(pred_score[0].cpu()*10.0))
                valid_exp_scores_list.extend(
                    np.array(true_scores[0].cpu()*10.0))
                valid_org_predictions_list.extend(
                    np.array(pred_score[1].cpu()*10.0))
                valid_org_scores_list.extend(
                    np.array(true_scores[1].cpu()*10.0))
                valid_cont_predictions_list.extend(
                    np.array(pred_score[2].cpu()*10.0))
                valid_cont_scores_list.extend(
                    np.array(true_scores[2].cpu()*10.0))
                valid_total_predictions_list.extend(
                    np.array(pred_score[3].cpu()*10.0))
                valid_total_scores_list.extend(
                    np.array(true_scores[3].cpu()*10.0))
                
            
            valid_exp_mse_mean = valid_exp_mse_mean/ii
            valid_exp_mae_mean = valid_exp_mae_mean/ii
            valid_org_mse_mean = valid_org_mse_mean/ii
            valid_org_mae_mean = valid_org_mae_mean/ii
            valid_cont_mse_mean = valid_cont_mse_mean/ii
            valid_cont_mae_mean = valid_cont_mae_mean/ii
            valid_total_mse_mean = valid_total_mse_mean/ii
            valid_total_mae_mean = valid_total_mae_mean/ii
            
            if not os.path.isdir("./result/cross_prompt"):
                os.mkdir("./result/cross_prompt")
            
            with open("./result/cross_prompt/exp_valid_pred.pkl", "wb") as f:
                pickle.dump(valid_exp_predictions_list, f)
            with open("./result/cross_prompt/exp_valid_true.pkl", "wb") as f:
                pickle.dump(valid_exp_scores_list, f)
            with open("./result/cross_prompt/org_valid_pred.pkl", "wb") as f:
                pickle.dump(valid_org_predictions_list, f)
            with open("./result/cross_prompt/org_valid_true.pkl", "wb") as f:
                pickle.dump(valid_org_scores_list, f)
            with open("./result/cross_prompt/cont_valid_pred.pkl", "wb") as f:
                pickle.dump(valid_cont_predictions_list, f)
            with open("./result/cross_prompt/cont_valid_true.pkl", "wb") as f:
                pickle.dump(valid_cont_scores_list, f)
            with open("./result/cross_prompt/total_valid_pred.pkl", "wb") as f:
                pickle.dump(valid_total_predictions_list, f)
            with open("./result/cross_prompt/total_valid_true.pkl", "wb") as f:
                pickle.dump(valid_total_scores_list, f)
            
            
            print("Epoch: {} | Train Exp RMSE: {: .4f} | Train Exp MAE: {: .4f} | Train Org RMSE: {: .4f} | Train Org MAE: {: .4f} | \
                Train Cont RMSE: {: .4f} | Train Cont MAE: {: .4f} | Train Total RMSE: {: .4f} | Train Total MAE: {: .4f} |\
                Valid Exp RMSE: {: .4f} | Valid Exp MAE: {: .4f} | Valid Org RMSE: {: .4f} | Valid Org MAE: {: .4f} | \
                Valid Cont RMSE: {: .4f} | Valid Cont MAE: {: .4f} | Valid Total RMSE: {: .4f} | Valid Total MAE: {: .4f} |"
                .format(epoch, np.sqrt(train_exp_mse_mean.cpu().detach()), train_exp_mae_mean.cpu().detach(), np.sqrt(train_org_mse_mean.cpu().detach()), 
                        train_org_mae_mean.cpu().detach(), np.sqrt(train_cont_mse_mean.cpu().detach()), train_cont_mae_mean.cpu().detach(),
                        np.sqrt(train_total_mse_mean.cpu().detach()), train_total_mae_mean.cpu().detach(), np.sqrt(valid_exp_mse_mean.cpu()), valid_exp_mae_mean.cpu().detach(),
                        np.sqrt(valid_org_mse_mean.cpu()), valid_org_mae_mean.cpu().detach(), np.sqrt(valid_cont_mse_mean.cpu()), valid_cont_mae_mean.cpu().detach(),
                        np.sqrt(valid_total_mse_mean.cpu()), valid_total_mae_mean.cpu().detach()))
            
            if early_stopper.should_stop(model, previous_score, valid_total_mse_mean):
                print("Early Stopping based on Valid MSE Loss")
                print(f"EarlyStopping: [Epoch: {epoch}]")
            else:
                th.save(model.state_dict(), os.path.join(
                    ckpt_path, "{}_model.ckpt".format(epoch)))
            previous_score = valid_total_mse_mean
            
    print("\n\nTraining and Validation have been successfully done!")

def model_test(test_data, test_dic, model, references, ckpt_path, args, prompt):
    
    print("Begin Testing...")
    
    if th.cuda.is_available() and args.gpu != -1:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"
    
    mse_fuction = nn.MSELoss()
    mae_function = nn.L1Loss()
    
    with th.no_grad():
        
        best_checkpoint = th.load(os.path.join(
            ckpt_path, "best_model_epoch.ckpt"
        ))
        
        model.load_state_dict(best_checkpoint["model_state_dict"])
        for param in model.pos_embedder.parameters():
            param.requires_grad = False
        print("The best model has been successfully loaded")
        test_exp_mse_mean = 0.0
        test_org_mse_mean = 0.0
        test_cont_mse_mean = 0.0
        test_total_mse_mean = 0.0
        test_exp_mae_mean = 0.0
        test_org_mae_mean = 0.0
        test_cont_mae_mean = 0.0
        test_total_mae_mean = 0.0
        test_exp_predictions_list = list()
        test_org_predictions_list = list()
        test_cont_predictions_list = list()
        test_total_predictions_list = list()
        test_exp_scores_list = list()
        test_org_scores_list = list()
        test_cont_scores_list = list()
        test_total_scores_list = list()
        test_sen_att = list()
        test_key_att = list()
        test_sim_sen = list()
        test_sim_key = list()
        test_sal_sen = list()
        test_sal_key = list()
        test_sen_pos_att = list()
        test_doc_pos_att = list()
        test_sim_sen_pos = list()
        test_sim_doc_pos = list()
        test_sal_sen_pos = list()
        test_sal_doc_pos = list()
        test_ids = list()
        test_ref_ids = list()
        for iii, test in enumerate(tqdm(test_data[prompt])):
            essay_data, scores = test
            id = essay_data[0]
            scoring_sen_embeddings = essay_data[1].to(device)
            scoring_key_embeddings = essay_data[2].to(device)
            prompt_ids = essay_data[3]
            scoring_sen_pos = essay_data[4].to(device)
            scoring_doc_pos = essay_data[5].to(device)
            ref_id = references[0]
            ref_sen_pos = references[1].to(device)
            ref_doc_pos = references[2].to(device)
            true_scores = scores.to(device)
            
            model.eval()
        
            pred_score, sim_sen, sim_key, sim_sen_pos, sim_doc_pos = model(prompt_ids, scoring_sen_embeddings, scoring_key_embeddings, 
                                                                           scoring_sen_pos, ref_sen_pos, scoring_doc_pos, ref_doc_pos, test=True)
            
            test_ids.append(id)
            test_ref_ids.append(ref_id)
            test_sen_att.append(sim_sen[2].cpu().detach())
            test_key_att.append(sim_key[2].cpu().detach())
            test_sim_sen.append(sim_sen[1].cpu().detach())
            test_sim_key.append(sim_key[1].cpu().detach())
            test_sal_sen.append(sim_sen[0].cpu().detach())
            test_sal_key.append(sim_key[0].cpu().detach())
            test_sen_pos_att.append(sim_sen_pos[2].cpu().detach())
            test_doc_pos_att.append(sim_doc_pos[2].cpu().detach())
            test_sim_sen_pos.append(sim_sen_pos[1].cpu().detach())
            test_sim_doc_pos.append(sim_doc_pos[1].cpu().detach())
            test_sal_sen_pos.append(sim_sen_pos[0].cpu().detach())
            test_sal_doc_pos.append(sim_doc_pos[0].cpu().detach())
            
            exp_loss = mse_fuction(pred_score[0], true_scores[:,0])
            org_loss = mse_fuction(pred_score[1], true_scores[:,1])
            cont_loss = mse_fuction(pred_score[2], true_scores[:,2])
            total_loss = mse_fuction(pred_score[3], true_scores[:,3])
            
            exp_mae = mae_function(pred_score[0], true_scores[:,0])
            org_mae = mae_function(pred_score[1], true_scores[:,1])
            cont_mae = mae_function(pred_score[2], true_scores[:,2])
            total_mae = mae_function(pred_score[3], true_scores[:,3])
            
            test_exp_mse_mean += exp_loss.data
            test_org_mse_mean += org_loss.data
            test_cont_mse_mean += cont_loss.data
            test_total_mse_mean += total_loss.data
            test_exp_mae_mean += exp_mae.data
            test_org_mae_mean += org_mae.data
            test_cont_mae_mean += cont_mae.data
            test_total_mae_mean += total_mae.data
            
            
            test_exp_predictions_list.extend(
                np.array(pred_score[0].cpu()*10.0))
            test_exp_scores_list.extend(
                np.array(true_scores[:,0].cpu()*10.0))
            test_org_predictions_list.extend(
                np.array(pred_score[1].cpu()*10.0))
            test_org_scores_list.extend(
                np.array(true_scores[:,1].cpu()*10.0))
            test_cont_predictions_list.extend(
                np.array(pred_score[2].cpu()*10.0))
            test_cont_scores_list.extend(
                np.array(true_scores[:,2].cpu()*10.0))
            test_total_predictions_list.extend(
                np.array(pred_score[3].cpu()*10.0))
            test_total_scores_list.extend(
                np.array(true_scores[:,3].cpu()*10.0))
        
            
        test_exp_mse_mean = test_exp_mse_mean/iii
        test_org_mse_mean = test_org_mse_mean/iii
        test_cont_mse_mean = test_cont_mse_mean/iii
        test_total_mse_mean = test_total_mse_mean/iii
        test_exp_mae_mean = test_exp_mae_mean/iii
        test_org_mae_mean = test_org_mae_mean/iii
        test_cont_mae_mean = test_cont_mae_mean/iii
        test_total_mae_mean = test_total_mae_mean/iii
        test_ids = th.cat(test_ids)
        test_ref_ids = th.cat(test_ref_ids)
        test_sen_att = th.cat(test_sen_att)
        test_key_att = th.cat(test_key_att)
        test_sim_sen = th.cat(test_sim_sen)
        test_sim_key = th.cat(test_sim_key)
        test_sen_pos_att = th.cat(test_sen_pos_att)
        test_doc_pos_att = th.cat(test_doc_pos_att)
        test_sim_sen_pos = th.cat(test_sim_sen_pos)
        test_sim_doc_pos = th.cat(test_sim_doc_pos)
        test_sal_sen = th.cat(test_sal_sen)
        test_sal_key = th.cat(test_sal_key)
        test_sal_sen_pos = th.cat(test_sal_sen_pos)
        test_sal_doc_pos = th.cat(test_sal_doc_pos)
        
        
        
        test_class_exp_loss = class_wise_mae_subscores(test_exp_scores_list, test_exp_predictions_list)
        test_class_org_loss = class_wise_mae_subscores(test_org_scores_list, test_org_predictions_list)
        test_class_cont_loss = class_wise_mae_subscores(test_cont_scores_list, test_cont_predictions_list)
        test_class_total_loss = class_wise_mae_subscores(test_total_scores_list, test_total_predictions_list)
        
        
        if not os.path.isdir("result/explainability/"):
            os.mkdir("result/explainability/")

        explain_path = os.path.join("result/explainability/", 'cross_prompt')
        if not os.path.isdir(explain_path):
            os.mkdir(explain_path)
        explain_path = os.path.join(explain_path, prompt)
        if not os.path.isdir(explain_path):
            os.mkdir(explain_path)
        
        
        with open("{}/test_ref_ids.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_ref_ids, f)
        with open("{}/test_sen_sal_sim.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sen_att, f)
        with open("{}/test_key_sal_sim.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_key_att, f)
        with open("{}/test_sim_sen.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sim_sen, f)
        with open("{}/test_sim_key.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sim_key, f)
        with open("{}/test_sal_sen.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sal_sen, f)
        with open("{}/test_sal_key.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sal_key, f)
        with open("{}/test_sentences.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_dic[prompt]["sentences"], f)
        with open("{}/test_keywords.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_dic[prompt]["keywords"], f)
        with open("{}/test_sen_pos_sal_sim.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sen_pos_att, f)
        with open("{}/test_doc_pos_sal_sim.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_doc_pos_att, f)
        with open("{}/test_sim_sen_pos.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sim_sen_pos, f)
        with open("{}/test_sim_doc_pos.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sim_doc_pos, f)
        with open("{}/test_sal_sen_pos.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sal_sen_pos, f)
        with open("{}/test_sal_doc_pos.pkl".format(explain_path), "wb") as f:
            pickle.dump(test_sal_doc_pos, f)
        
        
        with open("./result/cross_prompt/{}_test_exp_pred.pkl".format(prompt), "wb") as f:
            pickle.dump(test_exp_predictions_list, f)
        with open("./result/cross_prompt/{}_test_org_pred.pkl".format(prompt), "wb") as f:
            pickle.dump(test_org_predictions_list, f)
        with open("./result/cross_prompt/{}_test_cont_pred.pkl".format(prompt), "wb") as f:
            pickle.dump(test_cont_predictions_list, f)
        with open("./result/cross_prompt/{}_test_total_pred.pkl".format(prompt), "wb") as f:
            pickle.dump(test_total_predictions_list, f)
        with open("./result/cross_prompt/{}_test_exp_true.pkl".format(prompt), "wb") as f:
            pickle.dump(test_exp_scores_list, f)
        with open("./result/cross_prompt/{}_test_org_true.pkl".format(prompt), "wb") as f:
            pickle.dump(test_org_scores_list, f)
        with open("./result/cross_prompt/{}_test_cont_true.pkl".format(prompt), "wb") as f:
            pickle.dump(test_cont_scores_list, f)
        with open("./result/cross_prompt/{}_test_total_true.pkl".format(prompt), "wb") as f:
            pickle.dump(test_total_scores_list, f)
        with open("./result/cross_prompt/{}_exp_classwise.pkl".format(prompt), "wb") as f:
            pickle.dump(test_class_exp_loss, f)
        with open("./result/cross_prompt/{}_org_classwise.pkl".format(prompt), "wb") as f:
            pickle.dump(test_class_org_loss, f)
        with open("./result/cross_prompt/{}_cont_classwise.pkl".format(prompt), "wb") as f:
            pickle.dump(test_class_cont_loss, f)
        with open("./result/cross_prompt/{}_total_classwise.pkl".format(prompt), "wb") as f:
            pickle.dump(test_class_total_loss, f)
            
    print("Prompt : {} | Test Exp RMSE: {: .4f} | Test Org RMSE: {: .4f} | Test Cont RMSE: {: .4f} | Test Total RMSE: {: .4f} | \n\n\
        | Test Exp MAE: {: .4f} | Test Org MAE: {: .4f} | Test Cont MAE: {: .4f} | Test Total MAE: {: .4f} | \n\n\
        | Test Exp BAL LOSS: {: .4f} | Test Org BAL LOSS: {: .4f} | Test Cont BAL LOSS: {: .4f} | Test Total BAL LOSS: {: .4f} |".format(
        prompt, np.sqrt(test_exp_mse_mean.cpu()), np.sqrt(test_org_mse_mean.cpu()), np.sqrt(test_cont_mse_mean.cpu()), np.sqrt(test_total_mse_mean.cpu()), 
        test_exp_mae_mean.cpu(), test_org_mae_mean.cpu(), test_cont_mae_mean.cpu(), test_total_mae_mean.cpu(),
        np.mean(test_class_exp_loss), np.mean(test_class_org_loss), np.mean(test_class_cont_loss), np.mean(test_class_total_loss)))

    print("\n\nTesting has been successfully done!")
    
    return np.round(np.sqrt(test_exp_mse_mean.cpu().item()), 4), np.round(test_exp_mae_mean.cpu().item(), 4), np.round(np.mean(test_class_exp_loss).item(), 4), \
        np.round(np.sqrt(test_org_mse_mean.cpu().item()), 4), np.round(test_org_mae_mean.cpu().item(), 4), np.round(np.mean(test_class_org_loss).item(), 4), \
        np.round(np.sqrt(test_cont_mse_mean.cpu().item()), 4), np.round(test_cont_mae_mean.cpu().item(), 4), np.round(np.mean(test_class_cont_loss).item(), 4), \
        np.round(np.sqrt(test_total_mse_mean.cpu().item()), 4), np.round(test_total_mae_mean.cpu().item(), 4), np.round(np.mean(test_class_total_loss).item(), 4)
            
