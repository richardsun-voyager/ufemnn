import pickle
import numpy as np
import os
import copy
import torch

class data_generator:
    def __init__(self, config, data_path, is_training=True):
        '''
        Generate training and testing samples
        '''    
        self.is_training = is_training
        self.config = config
        self.index = 0
        self.data_batch = list(self.load_data(data_path))
        self.num_examples = len(self.data_batch)
        self.UNK = "unk"
        self.EOS = "<eos>"
        self.PAD = "<pad>"
        self.vocab_size = 0
        self.vocab, self.word2id, self.id2word = self.load_local_dict()
            
    def load_data(self, path):
        '''
        Load the pickle file
        '''
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


    def load_local_dict(self):
        '''
        Load dictionary files
        '''
        if not os.path.exists(self.config.dic_path):
            print('Dictionary file not exist!')
        with open(self.config.dic_path, 'rb') as f:
            vocab, word2id, id2word = pickle.load(f)
        self.vocab_size = len(vocab)
        self.UNK_ID = word2id[self.UNK]
        self.PAD_ID = word2id[self.PAD]
        self.EOS_ID = word2id[self.EOS]
        return vocab, word2id, id2word

    def generate_sample(self, data):
        '''
        Generate a batch of training dataset
        '''
        batch_size = self.config.batch_size
        select_index = np.random.choice(len(data), batch_size, replace=False)
        select_data = [data[i] for i in select_index]
        return select_data

    def shuffle_data(self):
        np.random.shuffle(self.data_batch)

    def reset_samples(self):
        self.index = 0

    def pad_data(self, sents, labels):
        '''
        Padding sentences to the same size
        '''
        sent_lens = [len(tokens) for tokens in sents]
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(labels)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)

        #padding sent with PAD IDs
        sent_vecs = np.ones([batch_size, max_len]) * self.PAD_ID
        sent_vecs = torch.LongTensor(sent_vecs)
        for i, s in enumerate(sents):#batch_size*max_len
            sent_vecs[i, :len(s)] = torch.LongTensor(s)
        sent_lens, perm_idx = sent_lens.sort(0, descending=True)
        sent_ids = sent_vecs[perm_idx]

        label_list = label_list[perm_idx]

        return sent_ids, label_list, sent_lens
            
    def get_ids_samples(self, is_balanced=False):
        '''
        Get samples including ids of words, labels
        '''
        if self.is_training:
            samples = self.generate_sample(self.data_batch)
            token_ids, label_list = zip(*samples)
            #Sorted according to the length
            sent_ids, label_list, sent_lens = self.pad_data(token_ids, label_list)
        else:
            if self.index == self.num_examples:
                print('Testing Over!')
            #First get batches of testing data
            if self.num_examples - self.index >= self.config.batch_size:
                #print('Testing Sample Index:', self.index)
                start = self.index
                end = start + self.config.batch_size
                samples = self.data_batch[start: end]
                self.index = end
                token_ids, label_list = zip(*samples)
                #Sorting happens here
                sent_ids,  label_list, sent_lens = self.pad_data(token_ids, label_list)

            else:#Then generate testing data one by one
                samples =  self.data_batch[self.index:] 
                if self.index == self.num_examples - 1:#if only one sample left
                    samples = [samples]
                token_ids, label_list = zip(*samples)
                sent_ids,  label_list, sent_lens = self.pad_data(token_ids, label_list)
                self.index += len(samples)
        yield sent_ids,  label_list, sent_lens


    def get_sequential_ids_samples(self, is_balanced=False):
        '''
        Get samples including ids of words, labels
        '''

        if self.index == self.num_examples:
            print('Testing Over!')
        #First get batches of testing data
        if self.num_examples - self.index >= self.config.batch_size:
            #print('Testing Sample Index:', self.index)
            start = self.index
            end = start + self.config.batch_size
            samples = self.data_batch[start: end]
            self.index = end
            token_ids, label_list = zip(*samples)
            #Sorting happens here
            sent_ids, label_list, sent_lens = self.pad_data(token_ids, label_list)

        else:#Then generate testing data one by one
            samples =  self.data_batch[self.index:]
            if self.index == self.num_examples - 1:#if only one sample left
                samples = [samples]
            token_ids, label_list = zip(*samples)
            sent_ids, label_list, sent_lens = self.pad_data(token_ids, label_list)
            self.index += len(samples)
        yield sent_ids, label_list, sent_lens

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask
