import os
import json
import math
import numpy as np
from random import choice

from  preprocess import split_dataset
from tokenizer import OurTokenizer, read_token

maxlen = 160


class data_generator:

    def __init__(self, data, tokenizer, predicate2id,batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps =  int(math.ceil(len(self.data) / self.batch_size))
        self.tokenizer = tokenizer
        self.predicate2id = predicate2id
        self.num_classes = len(predicate2id)

    def __len__(self):
        return self.steps

    def __iter__(self):

        def seq_padding(X, padding=0):
            L = [len(x) for x in X]
            ML = max(L)
            return np.array([
                np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
            ])

        def list_find(list1, list2):
            n_list2 = len(list2)
            for i in range(len(list1)):
                if list1[i: i+n_list2] == list2:
                    return i
            return -1

        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T1, T2, S1, S2, K1, K2, O1, O2 = [], [], [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d['text'][:maxlen]
                tokens = self.tokenizer.tokenize(text)
                items = {}
                for sp in d['spo_list']:
                    sp = (self.tokenizer.tokenize(sp[0])[1:-1], sp[1], self.tokenizer.tokenize(sp[2])[1:-1])
                    subjectid = list_find(tokens, sp[0])
                    objectid = list_find(tokens, sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid+len(sp[0]))
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid , objectid+len(sp[2]),
                                           self.predicate2id[sp[1]]))
                if items:
                    t1, t2 = self.tokenizer.encode(first=text)
                    T1.append(t1)
                    T2.append(t2)
                    s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1]-1] = 1
                    k1, k2 = np.array(list(items.keys())).T 
                    k1 = choice(k1)         
                    k2 = choice(k2[k2 >= k1])  
                    o1, o2 = np.zeros((len(tokens), self.num_classes)), np.zeros((len(tokens), self.num_classes))
                    for j in items.get((k1, k2), []):
                        o1[j[0]][j[2]] = 1
                        o2[j[1]-1][j[2]] = 1
                    S1.append(s1)       
                    S2.append(s2)       
                    K1.append([k1])     
                    K2.append([k2-1])   
                    O1.append(o1)       
                    O2.append(o2)       

                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)
                        T2 = seq_padding(T2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        O1 = seq_padding(O1, np.zeros(self.num_classes))
                        O2 = seq_padding(O2, np.zeros(self.num_classes))
                        K1, K2 = np.array(K1), np.array(K2)
                        yield [T1, T2, S1, S2, K1, K2, O1, O2], None
                        T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []


if __name__ == '__main__':
    bert_vocab_path = os.path.join(os.path.dirname(__file__),'..','bert','vocab.txt')
    token_dict = read_token(bert_vocab_path)
    tokenizer = OurTokenizer(token_dict)

    train_file = os.path.join(os.path.dirname(__file__),'datasets','train_data_me.json')
    dev_file = os.path.join(os.path.dirname(__file__),'datasets','dev_data_me.json')
    schema_file = os.path.join(os.path.dirname(__file__),'datasets', 'all_50_schemas_me.json')

    train_data = json.load(open(train_file))
    dev_data = json.load(open(dev_file))
    _, predicate2id = json.load(open(schema_file))

    total_data = []
    total_data.extend(train_data)
    total_data.extend(dev_data)

    train_data,test_data = split_dataset(total_data)

    gen = data_generator(train_data,tokenizer,predicate2id).__iter__()
    print(next(gen))