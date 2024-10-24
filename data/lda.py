import json
import numpy as np
import torch

from toptoolkit.pytorch.data import get_dataloader


class LDAConfig:
    start_size = 2
    word_batch_size = 100
    online_it = 2
    num_active_per_it = 10
    num_eval = 50


def get_lda(split, batch_size, train=True):
    if split not in ['1k', '3k', '5k']:
        raise NotImplementedError
    
    with open(f'data/lda_jsons/lda_top_words_split_allwords_{split}.json', 'r') as f:
        content = json.load(f)

    # data = {}
    # for partition in ['train', 'test', 'val']:
    #     data[partition] = torch.LongTensor(content['data'][partition])
    partition = 'train' if train else 'test'
    data = torch.LongTensor(content['data'][partition])

    ind2word = {int(k): v for k, v in content['params']['ind2word'].items()}
    word2ind = content['params']['word2ind']

    dataloader = get_dataloader(data, batch_size, train)
    return dataloader, ind2word, word2ind


def get_lda_mini(batch_size):
    with open('data/lda_jsons/lda_top_words_split_allwords_1k.json', 'r') as f:
        content = json.load(f)

    np_data = np.asarray(content['data']['test'][:20]).reshape(-1)
    # np_data = np.asarray(content['data']['test'][:2]).reshape(-1)

    ind2ind = {}
    for x in np_data:
        if x not in ind2ind:
            ind2ind[x] = len(ind2ind)

    old_ind2word = {int(k): v for k, v in content['params']['ind2word'].items()}
    old_word2ind = content['params']['word2ind']
    ind2word = {}
    word2ind = {}
    for k, v in old_ind2word.items():
        if k in ind2ind:
            ind2word[ind2ind[k]] = v
    for k, v in old_word2ind.items():
        if v in ind2ind:
            word2ind[k] = ind2ind[v]
    for i in range(len(np_data)):
        np_data[i] = ind2ind[np_data[i]]
    
    data = torch.LongTensor(np_data.reshape(20, -1))
    # data = torch.LongTensor(np_data.reshape(2, -1))

    dataloader = get_dataloader(data, batch_size, False)
    return dataloader, ind2word, word2ind