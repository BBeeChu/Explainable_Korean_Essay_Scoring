import os
import argparse
import json
import yaml
import configparser
import torch as th
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError
from utils import load_base_config, cross_prompt_load_data, MyDataset, EarlyStopping, model_train, model_test
from loss import class_wise_mae_subscores
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam, AdamW
import torch.optim as optim
from model import Saliency_Essay_Model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import math
import random



def main(args, train_prompt, test_prompt, result_dic, test=False):
    
    if not test:
        print("Train & Test")
    else:
        print("Test Only")
    random.seed(args.seed)

    th.manual_seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    th.cuda.manual_seed(args.seed)
    
    
    
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", args.task)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
        
    

    if th.cuda.is_available() and args.gpu != -1:
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"
    
    WEIGHT_PATH = './pos_weights/pos_weights.ckpt'
    PATH = "./data/total_dataset.csv"
    preprocessed_data, references, mean_score, train_indices, test_index_list, \
        prompt_sen_embeddings, prompt_key_embeddings = cross_prompt_load_data(PATH, args, train_prompt, test_prompt)
    
    
    val_sizes = list()
    for index in test_index_list:
        val_sizes.append(math.ceil(len(index)/2))
    
    train_dic = dict()
    train_dic["ids"] = preprocessed_data[0][train_indices]
    train_dic["sentence_embeddings"] = preprocessed_data[1][train_indices]
    train_dic["keyword_embeddings"] = preprocessed_data[2][train_indices]
    train_dic["scores"] = preprocessed_data[3][train_indices]
    train_dic["sentences"] = preprocessed_data[4][train_indices]
    train_dic["keywords"] = preprocessed_data[5][train_indices]
    train_dic["prompt_ids"] = preprocessed_data[6][train_indices]
    train_dic["sen_pos"] = preprocessed_data[7][train_indices]
    train_dic["doc_pos"] = preprocessed_data[8][train_indices]
    
    val_ids = list()
    val_sen_embeds = list()
    val_key_embeds = list()
    val_scores = list()
    val_sens = list()
    val_keys = list()
    val_prompt_ids = list()
    val_sen_pos = list()
    val_doc_pos = list()
    for i in range(len(test_prompt)):
        val_ids.append(preprocessed_data[0][test_index_list[i]][:val_sizes[i]])
        val_sen_embeds.append(preprocessed_data[1][test_index_list[i]][:val_sizes[i]])
        val_key_embeds.append(preprocessed_data[2][test_index_list[i]][:val_sizes[i]])
        val_scores.append(preprocessed_data[3][test_index_list[i]][:val_sizes[i]])
        val_sens.append(preprocessed_data[4][test_index_list[i]][:val_sizes[i]])
        val_keys.append(preprocessed_data[5][test_index_list[i]][:val_sizes[i]])
        val_prompt_ids.append(preprocessed_data[6][test_index_list[i]][:val_sizes[i]])
        val_sen_pos.append(preprocessed_data[7][test_index_list[i]][:val_sizes[i]])
        val_doc_pos.append(preprocessed_data[8][test_index_list[i]][:val_sizes[i]])
    
    val_ids = np.concatenate(val_ids, axis=0)
    val_sentence_embeddings = np.concatenate(val_sen_embeds, axis=0)
    val_keyword_embeddings = np.concatenate(val_key_embeds, axis=0)
    val_sentences = np.concatenate(val_sens, axis=0)
    val_keywords = np.concatenate(val_keys, axis=0)
    val_prompt_ids = np.concatenate(val_prompt_ids, axis=0)
    val_sen_pos = np.concatenate(val_sen_pos, axis=0)
    val_doc_pos = np.concatenate(val_doc_pos, axis=0)
    val_scores = np.concatenate(val_scores, axis=0)
    
    val_dic = dict()
    val_dic["ids"] = val_ids
    val_dic["sentence_embeddings"] = val_sentence_embeddings
    val_dic["keyword_embeddings"] = val_keyword_embeddings
    val_dic["sentences"] = val_sentences
    val_dic["keywords"] = val_keywords
    val_dic["prompt_ids"] = val_prompt_ids
    val_dic["sen_pos"] = val_sen_pos
    val_dic["doc_pos"] = val_doc_pos
    val_dic["scores"] = val_scores
    
    test_dic = dict()
    for p in test_prompt:
        test_dic[p] = dict()
    for i, p in enumerate(test_prompt):
        test_dic[p]["ids"] = preprocessed_data[0][test_index_list[i]][val_sizes[i]:]
        test_dic[p]["sentence_embeddings"] = preprocessed_data[1][test_index_list[i]][val_sizes[i]:]
        test_dic[p]["keyword_embeddings"] = preprocessed_data[2][test_index_list[i]][val_sizes[i]:]
        test_dic[p]["scores"] = preprocessed_data[3][test_index_list[i]][val_sizes[i]:]
        test_dic[p]["sentences"] = preprocessed_data[4][test_index_list[i]][val_sizes[i]:]
        test_dic[p]["keywords"] = preprocessed_data[5][test_index_list[i]][val_sizes[i]:]
        test_dic[p]["prompt_ids"] = preprocessed_data[6][test_index_list[i]][val_sizes[i]:]
        test_dic[p]["sen_pos"] = preprocessed_data[7][test_index_list[i]][val_sizes[i]:]
        test_dic[p]["doc_pos"] = preprocessed_data[8][test_index_list[i]][val_sizes[i]:]
    
    num_keyword = train_dic["keyword_embeddings"].shape[1]
    num_sentence = train_dic["sentence_embeddings"].shape[1]
    
    
    train_dataset = MyDataset(train_dic)
    
    val_dataset = MyDataset(val_dic)
    
    test_dataset = dict()
    for p in test_prompt:
        test_dataset[p] = MyDataset(test_dic[p])
    
    
    train_data = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
    )
    
    valid_data = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    test_data = dict()
    for p in test_prompt:
        test_data[p] = DataLoader(test_dataset[p], batch_size=args.batch_size, shuffle=True)
    
    prompt_sen_embeddings = prompt_sen_embeddings.to(device)
    prompt_key_embeddings = prompt_key_embeddings.to(device)
    model = Saliency_Essay_Model(args, mean_score, num_keyword, num_sentence, prompt_sen_embeddings, prompt_key_embeddings).to(device)
    pos_weight = th.load(WEIGHT_PATH)
    model.pos_embedder.load_state_dict(pos_weight['model_state_dict'])
    
    if not test:
        model_train(train_data, valid_data, model, references, ckpt_path, args)
    
    for prompt in test_prompt:
        exp_rmse, exp_mae, exp_bal, org_rmse, org_mae, org_bal, cont_rmse, cont_mae, \
        cont_bal, total_rmse, total_mae, total_bal = model_test(test_data, test_dic, model, references, ckpt_path, args, prompt)

        result_dic['prompt'].append(prompt)
        result_dic['exp_rmse'].append(exp_rmse)
        result_dic['org_rmse'].append(org_rmse)
        result_dic['cont_rmse'].append(cont_rmse)
        result_dic['total_rmse'].append(total_rmse)
        result_dic['exp_mae'].append(exp_mae)
        result_dic['org_mae'].append(org_mae)
        result_dic['cont_mae'].append(cont_mae)
        result_dic['total_mae'].append(total_mae)
        result_dic['exp_bal'].append(exp_bal)
        result_dic['org_bal'].append(org_bal)
        result_dic['cont_bal'].append(cont_bal)
        result_dic['total_bal'].append(total_bal)
    
    return result_dic
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Essay Scoring')
    parser.add_argument('--task', '-sl', type=str, default='cross_prompt', help='level of school')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='which gpu to use, specify -1 to use CPU')
    parser.add_argument('--config', '-c', type=str, help='config file for model hyperparameters')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch_size')
    parser.add_argument('--optimizer', '-o', type=str, default="adam", help='the specific name of optimizer')
    parser.add_argument('--seed', '-s', type=int, default=42, help='random seed')
    parser.add_argument('--num_epochs', '-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--hidden_dim', '-d', type=int, default=100, help='hidden dimension')
    parser.add_argument('--num_ref', '-r', type=int, default=20, help='number of reference essays')
    parser.add_argument('--prompt', '-pr', type=str, default='미래 도시에 대한 본인의 생각', help='prompt for essays')
    parser.add_argument('--patience', '-p', type=int, default=5, help='number of patience for early stopping')
    parser.add_argument('--num_sentence', '-ns', type=int, default=10, help='number of sentences')
    parser.add_argument('--num_keyword', '-nk', type=int, default=10, help='number of keywords')
    parser.add_argument('--activation_function', '-af', type=str, default='relu', help='activation function')
    parser.add_argument('--measure', '-m', type=str, default='total_score', help='measure to assess')
    parser.add_argument('--loss', '-l', type=str, default='both', help='training manner (e.g. both, mse, mae)')
    
    args = parser.parse_args()
    if args.config is None:
        args.config = "./configs.json"

    configs = load_base_config()
    configs.update(vars(args))
    args = argparse.Namespace(**configs)
    
    
    measure_list = ['total_exp', 'total_cont', 'total_org', 'total_score']
   
    
    train_prompt = ['미래의 나의 모습', '표절을 해결할 수 있는 대안제시', '학교 폭력에 대한 본인 생각 및 해결방안을 작성',
                    '본인의 성격', '책임감을 느꼈던 사례', '습관들이기 위한 노력과 경험', '국내 여행', '전통과 악습에 대한 본인 생각을 작성',
                    'SNS상의 문제에 대한 본인의 생각 작성', '생물학적으로 다른 남/여에 대한 본인의 생각 작성', '동물 사육에 대한 본인 의견을 작성',
                    '사회적 불평등에 대한 본인의 생각 작성']
    
    test_prompt = ['미래 도시에 대한 본인의 생각', '한류를 지키는 방법', '참된 스승', '이상형과 나의 노력',
                   'e스포츠에 대한 본인 생각 작성', '카피레프트 운동에 대한 본인의 의견을 작성']
        

    result_dic = dict()
    result_dic['prompt'] = []
    result_dic['exp_rmse'] = []
    result_dic['org_rmse'] = []
    result_dic['cont_rmse'] = []
    result_dic['total_rmse'] = []
    result_dic['exp_mae'] = []
    result_dic['org_mae'] = []
    result_dic['cont_mae'] = []
    result_dic['total_mae'] = []
    result_dic['exp_bal'] = []
    result_dic['org_bal'] = []
    result_dic['cont_bal'] = []
    result_dic['total_bal'] = []
    
    
            
    df_dict = main(args, train_prompt, test_prompt, result_dic)
    
    
    result_df = pd.DataFrame(df_dict)
    result_df.to_csv("./result/cross_prompt_result_df.csv", index=False)


    