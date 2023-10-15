import sys
sys.path.append("..")
import os
from modules.mlp_model import MLP_Model
from modules.attention_model import Attention_Model
from modules.cnn_model import CNN_Model
from utils import data_generator
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import random
from argparse import ArgumentParser
from tqdm import tqdm
import pickle
import en_core_web_sm
nlp = en_core_web_sm.load()

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def evaluate(data_generator, model):
    device = torch.device(model.device)
    model.eval()
    count = 0
    gold_labels = []
    pred_labels = []
    data_generator.reset_samples()
    while data_generator.index < data_generator.num_examples:
        sent_ids, label_list, sent_lens = next(data_generator.get_ids_samples())
        output, _ = model(sent_ids.to(device), sent_lens.to(device))
        if model.label_dim == 1:
            preds = (output>0.5).squeeze()
        else:
            preds = output.argmax(dim=1, keepdim=True).squeeze()
        gold_labels += list(label_list.cpu().numpy())
        pred_labels += list(preds.cpu().numpy())
        num = (preds.cpu() == label_list.bool()).sum().cpu().item()
        count += num
    accuracy = count*1.0/data_generator.num_examples
    return accuracy

def get_dataset_path(dataset):
    train_path = './ntk/data/' + dataset + '/train_seq.pkl'
    dev_path = './ntk/data/' + dataset + '/dev_seq.pkl'
    test_path = './ntk/data/' + dataset + '/test_seq.pkl'
    dic_path = './ntk/data/' + dataset + '/vocab/local_dict.pkl'
    return train_path, dev_path, test_path, dic_path

def get_pos_neg_tokens(dataset_name):
    import pandas as pd 
    imdb_train_path = f'./ntk/data/{dataset_name}/train.csv'
    imdb_train = pd.read_csv(imdb_train_path)
    imdb_train_text = list(imdb_train.text.values)
    imdb_train_label = list(imdb_train.label.values)
    
    vocab = set()
    token_label_pair = []   # a list of (token, label) tuples
    token_doc_freq = {}     # {word: frequency}

    for i, text in enumerate(imdb_train_text):
        text = [w.text.lower() for w in nlp.tokenizer(text)]
        label = imdb_train_label[i]
        for w in text:
            vocab.add(w)
            token_doc_freq[w] = token_doc_freq.get(w, 0) + 1
            token_label_pair += [(w, label)]
    
    from collections import Counter
    token_label_pair_freq = Counter(token_label_pair)
    token_freq_pos = {}
    token_freq_neg = {}
    pos_tokens_freq = {}
    neg_tokens_freq = {}
    neutral_tokens_freq = {}
    
    if dataset_name == 'imdb':
        freq_threshold = 50
    elif dataset_name == 'sst':
        freq_threshold = 5
    for w in vocab:
        freq_pos = token_label_pair_freq[(w, 1)]
        freq_neg = token_label_pair_freq[(w, 0)]
        token_freq_pos[w] = freq_pos
        token_freq_neg[w] = freq_neg

        if freq_pos-freq_neg != 0:
            #We use 10
            rate = (freq_pos-freq_neg)/(freq_pos+freq_neg)*1.0
            if rate>0.5 and freq_pos>freq_threshold:#positive tokens
                pos_tokens_freq[w] = freq_pos
            if rate<-0.5 and freq_neg>freq_threshold:
                neg_tokens_freq[w] = freq_neg

        if abs(freq_pos-freq_neg)/(freq_pos+freq_neg)<0.1 and abs(freq_pos-freq_neg)<5:
            neutral_tokens_freq[w] = freq_neg + freq_pos

    from collections import OrderedDict
    pos_tokens_freq_sorted = OrderedDict(sorted(pos_tokens_freq.items(), key=lambda item: item[1], reverse=True))
    neg_tokens_freq_sorted = OrderedDict(sorted(neg_tokens_freq.items(), key=lambda item: item[1], reverse=True))
    neutral_tokens_freq_sorted = OrderedDict(sorted(neutral_tokens_freq.items(), key=lambda item: item[1], reverse=True))

    pos_tokens = list(pos_tokens_freq_sorted.keys())
    neg_tokens = list(neg_tokens_freq_sorted.keys())
    neutral_tokens = list(neutral_tokens_freq_sorted.keys())
    return pos_tokens, neg_tokens, neutral_tokens


def calculate_polarity_mlp(model, pos_tokens, neg_tokens, neutral_tokens):
    polarity = torch.matmul(model.embeddings.weight.data.detach(), model.hidden.weight.data.T.detach())
    if model.activation == 'relu':
        polarity = torch.relu(polarity)
    elif model.activation == 'tanh':
        polarity = torch.tanh(polarity)
        polarity = polarity
    polarity = torch.matmul(polarity, model.final.weight.data.T.detach())[:, 0].squeeze()
    polarity = polarity / model.hidden_dim      # [vocab_size, 1]
    pos_polarity = polarity[pos_tokens]
    neg_polarity = polarity[neg_tokens]
    neutral_polarity = polarity[neutral_tokens]
    return pos_polarity, neg_polarity, neutral_polarity

def calculate_polarity_attn(model, pos_token_ids, neg_token_ids, neutral_token_ids):
    final_weights = model.final.weight.data.detach()
    embs_weights = model.embeddings.weight.data.detach()
    polarity_scores = final_weights.matmul(embs_weights.transpose(0,1)).squeeze() / model.hidden_dim
    return polarity_scores[pos_token_ids], polarity_scores[neg_token_ids], polarity_scores[neutral_token_ids]
   
def calculate_polarity_cnn(model, pos_token_ids, neg_token_ids, neutral_token_ids):
    embs_weights = model.embeddings.weight.data.detach()
    cnn_weights = model.cnn.weight.data.detach()
    cnn1 = cnn_weights[:, :, 0]
    cnn2 = cnn_weights[:, :, 1]
    cnn3 = cnn_weights[:, :, 2]
    if model.activation == 'relu':
        polarity1 = model.final(torch.relu(cnn1.matmul(embs_weights.transpose(0, 1))).transpose(0, 1)).detach().squeeze() / model.hidden_dim
        polarity2 = model.final(torch.relu(cnn2.matmul(embs_weights.transpose(0, 1))).transpose(0, 1)).detach().squeeze() / model.hidden_dim
        polarity3 = model.final(torch.relu(cnn3.matmul(embs_weights.transpose(0, 1))).transpose(0, 1)).detach().squeeze() / model.hidden_dim
    elif model.activation == 'tanh':
        polarity1 = model.final(torch.tanh(cnn1.matmul(embs_weights.transpose(0, 1))).transpose(0, 1)).detach().squeeze() / model.hidden_dim
        polarity2 = model.final(torch.tanh(cnn2.matmul(embs_weights.transpose(0, 1))).transpose(0, 1)).detach().squeeze() / model.hidden_dim
        polarity3 = model.final(torch.tanh(cnn3.matmul(embs_weights.transpose(0, 1))).transpose(0, 1)).detach().squeeze() / model.hidden_dim
    pos_polarity = polarity1[pos_token_ids], polarity2[pos_token_ids], polarity3[pos_token_ids]
    neg_polarity = polarity1[neg_token_ids], polarity2[neg_token_ids], polarity3[neg_token_ids]
    neutral_polarity = polarity1[neutral_token_ids], polarity2[pos_token_ids], polarity3[pos_token_ids]
    return pos_polarity, neg_polarity, neutral_polarity

def train(config):

    device = torch.device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    config.train_path, config.dev_path, config.test_path, config.dic_path = get_dataset_path(config.dataset)

    train_dg = data_generator(config, config.train_path)
    train_eval_dg = data_generator(config, config.train_path, False)
    dev_dg = data_generator(config, config.dev_path, False)
    test_dg = data_generator(config, config.test_path, False)
    config.vocab_size = train_dg.vocab_size
    config.num_train = train_dg.num_examples

    loop_num = int(train_dg.num_examples/config.batch_size) + 1

    if 'mlp' in config.model_name:
        model = MLP_Model(config)
    elif 'attention' in config.model_name:
        model = Attention_Model(config)
    elif 'cnn' in config.model_name:
        model = CNN_Model(config)

    model.to(config.device)
    if config.label_dim == 1:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    best_dev_acc = -0.1
    best_model_test_acc = -0.1
    best_epoch = -1
    start_time = time.time()

    pos_tokens, neg_tokens, neutral_tokens = get_pos_neg_tokens(config.dataset)
    pos_token_ids = []
    neg_token_ids = []
    neutral_token_ids = []
    for w in pos_tokens:
        pos_token_ids.append(train_dg.word2id[w])
    for w in neg_tokens:
        neg_token_ids.append(train_dg.word2id[w])
    for w in neutral_tokens:
        neutral_token_ids.append(train_dg.word2id[w])

    polarity_scores = []

    for epoch in range(config.epochs):
        # Calculate NTK for each epoch

        total_loss = 0.0
        model.train()
        train_dg.shuffle_data()     
        train_dg.reset_samples()  

        for j in range(loop_num):
            x, labels, sent_lens = next(train_dg.get_sequential_ids_samples())
            optimizer.zero_grad()
            probs, outputs = model(x.to(config.device), sent_lens.to(config.device))

            labels = labels.to(device)
            if config.label_dim == 1:
                loss = criterion(probs.squeeze(), labels.float())
            else:
                loss = criterion(probs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_acc = evaluate(train_eval_dg, model)
        dev_acc = evaluate(dev_dg, model)
        test_acc = evaluate(test_dg, model)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_model_test_acc = test_acc
            best_epoch = epoch
            # torch.save(model.state_dict(), config.model_dir + config.save_name + config.hidden_dim)

        print(f"{now()} Epoch{epoch}: loss: {total_loss}, train_acc: {train_acc}, dev_acc: {dev_acc}, test_acc: {test_acc}")
        if 'mlp' in config.model_name:
            polarity_epoch = calculate_polarity_mlp(model, pos_token_ids, neg_token_ids, neutral_token_ids)
        elif 'attention' in config.model_name:
            polarity_epoch = calculate_polarity_attn(model, pos_token_ids, neg_token_ids, neutral_token_ids)
        elif 'cnn' in config.model_name:
            polarity_epoch = calculate_polarity_cnn(model, pos_token_ids, neg_token_ids, neutral_token_ids)
        polarity_scores.append(polarity_epoch)

    save_name = config.model_name + config.dataset + str(config.hidden_dim) + config.activation + str(config.final_init) 

    # Save the polarity scores
    if not os.path.exists('./polarity_scores'):
        os.makedirs('./polarity_scores')
    with open('./polarity_scores/' + save_name, "wb") as outfile:
        pickle.dump(polarity_scores, outfile)

    end_time = time.time()
    print("*"*20)
    print(f"{now()} finished; epoch {best_epoch} best_acc: {best_dev_acc} best_model_test_acc: {best_model_test_acc}, time/epoch: {(end_time-start_time)/config.epochs}")
    return

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of the embedding')
    parser.add_argument('--label_dim', type=int, default=1, help='label dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--model_dir', type=str, default='../affine/params_affine/', help='directory that saves the trained parameters')
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--dataset', type=str, default='sst', help='[sst, 20news, 20news_3classes, imdb]')
    parser.add_argument('--model_name', type=str, default='mlp', help='mlp, attention or cnn')
    parser.add_argument('--activation', type=str, default='relu', help='relu, tanh') 
    parser.add_argument('--final_init', type=float, default=0.1, help='final layer initialization')

    config = parser.parse_args()
    print(config)
    
    train(config)