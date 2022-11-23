import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import logging
import os
import config
from util import Lang


class Mydata(data.Dataset):
    def __init__(self, src_seq, trg_seq, length, src_word2id, trg_word2id):
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.length = length
        self.max_len = max(length)

    def __getitem__(self, index):
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, self.length[index], 'PAD')
        trg_seq = self.preprocess(trg_seq, self.trg_word2id, self.length[index], 'O')
        return src_seq, trg_seq, self.length[index]

    def __len__(self):
        return len(self.src_seqs)

    def preprocess(self, sequence, word2id, size, pad_name):
        """Converts words to ids."""
        sequence = [word2id[word] if word in word2id else word2id['UNK'] for word in sequence]
        sequence = torch.tensor(sequence)
        # padded_seqs = torch.ones(self.max_len).long()
        padded_seqs = torch.full((self.max_len,), word2id[pad_name]).long()
        padded_seqs[:size] = sequence
        return padded_seqs


def collate_fn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: x[2], reverse=True)
    # separate source and target sequences
    src_seqs, trg_seqs, size = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    max_len = size[0]
    src_seqs = [seq[:max_len].tolist() for seq in src_seqs]
    trg_seqs = [seq[:max_len].tolist() for seq in trg_seqs]
    src_seqs = torch.tensor(src_seqs).to(config.device)
    trg_seqs = torch.tensor(trg_seqs).to(config.device)
    return src_seqs, trg_seqs, size


def get_data(corpus, label, lang, batch_size, train):
    length = []
    x_seq = []
    y_seq = []
    for index, (x, y) in enumerate(zip(corpus, label)):
        if train:
            lang.index_words(x)
            lang.index_tags(y)
        x_seq.append(x.split())
        y_seq.append(y.split())
        assert len(x_seq[index]) == len(y_seq[index])
        length.append(len(x_seq[index]))
    dataset = Mydata(src_seq=x_seq,
                     trg_seq=y_seq,
                     length=length,
                     src_word2id=lang.word2index,
                     trg_word2id=lang.tag2index)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


def prepare_data(data_dir, filename, lang, batch_size, split=True):
    corpus = open(os.path.join(data_dir, filename + '_corpus.txt'), encoding='utf-8').readlines()
    labels = open(os.path.join(data_dir, filename + '_label.txt'), encoding='utf-8').readlines()
    if split:
        train_X, valid_X, train_y, valid_y = train_test_split(corpus, labels, test_size=0.2, random_state=0)
        print("Train: Number: {}".format(len(train_X)))
        logging.info("Train: Number: {}".format(len(train_X)))
        print("Valid: Number: {}".format(len(valid_X)))
        logging.info("Valid: Number: {}".format(len(valid_X)))
        train = get_data(train_X, train_y, lang, batch_size, True)
        valid = get_data(valid_X, valid_y, lang, batch_size, True)
        return train, valid
    else:
        print("Test: Number: {}".format(len(corpus)))
        logging.info("Test: Number: {}".format(len(corpus)))
        test = get_data(corpus, labels, lang, batch_size, False)
        return test


# if __name__ == '__main__':
#     lang = Lang()
#     train, dev = prepare_data(config.data_dir, config.train_name, lang, config.batch_size)
#     for x, y, size in train:
#         print("x: ", x.shape)
#         print("y: ", y.shape)
#         print("size: ", size)
#         break