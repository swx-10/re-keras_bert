#! -*- coding:utf-8 -*-

import re, os, json
import numpy as np
from tqdm import tqdm
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

from preprocess import split_dataset, repair
from generator import data_generator
from tokenizer import OurTokenizer, read_token

# 关闭warning提示
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

mode = 0
# 学习率
learning_rate = 5e-5
# 最小学习率
min_learning_rate = 1e-5

# bert模型及配置文件
config_path = r'E:\pretrain_weights\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'E:\pretrain_weights\chinese_L-12_H-768_A-12\bert_model.ckpt'
dict_path = r'E:\pretrain_weights\chinese_L-12_H-768_A-12\vocab.txt'

# 训练文件
train_file = os.path.join(os.path.dirname(__file__),'datasets','train_data_me.json')
# 开发文件
dev_file = os.path.join(os.path.dirname(__file__),'datasets','dev_data_me.json')
# 类别文件
schema_file = os.path.join(os.path.dirname(__file__),'datasets', 'all_50_schemas_me.json')

# 加载json语料数据
train_data = json.load(open(train_file))
dev_data = json.load(open(dev_file))
id2predicate, predicate2id = json.load(open(schema_file))
id2predicate = {int(i):j for i,j in id2predicate.items()}
num_classes = len(id2predicate)

# 完整数据集
total_data = []
total_data.extend(train_data)
total_data.extend(dev_data)

# 拆分train和test数据集
train_data, test_data = split_dataset(total_data, split_rate=0.8)
# 拆分train和valid数据集
train_data, valid_data = split_dataset(train_data, split_rate=0.9)

predicates = {} 

for d in train_data:
    repair(d)
    for sp in d['spo_list']:
        if sp[1] not in predicates:
            predicates[sp[1]] = []
        predicates[sp[1]].append(sp)
for d in dev_data:
    repair(d)

token_dict = read_token(dict_path)
tokenizer = OurTokenizer(token_dict)

def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return tf.gather_nd(seq, idxs)

def create_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    t1_in = Input(shape=(None,))
    t2_in = Input(shape=(None,))
    s1_in = Input(shape=(None,))
    s2_in = Input(shape=(None,))
    k1_in = Input(shape=(1,))
    k2_in = Input(shape=(1,))
    o1_in = Input(shape=(None, num_classes))
    o2_in = Input(shape=(None, num_classes))

    t1, t2, s1, s2, k1, k2, o1, o2 = t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in

    mask = Lambda(
        lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t1)

    t = bert_model([t1, t2])
    ps1 = Dense(1, activation='sigmoid')(t)
    ps2 = Dense(1, activation='sigmoid')(t)

    subject_model = Model([t1_in, t2_in], [ps1, ps2])

    k1v = Lambda(seq_gather)([t, k1])
    k2v = Lambda(seq_gather)([t, k2])

    kv = Average()([k1v, k2v])
    t = Add()([t, kv])
    po1 = Dense(num_classes, activation='sigmoid')(t)
    po2 = Dense(num_classes, activation='sigmoid')(t)

    object_model = Model([t1_in, t2_in, k1_in, k2_in], [po1, po2]) 

    train_model = Model([t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in],
                        [ps1, ps2, po1, po2])

    s1 = K.expand_dims(s1, 2)
    s2 = K.expand_dims(s2, 2)

    s1_loss = K.binary_crossentropy(s1, ps1)
    s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
    s2_loss = K.binary_crossentropy(s2, ps2)
    s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

    o1_loss = K.sum(K.binary_crossentropy(o1, po1), 2, keepdims=True)
    o1_loss = K.sum(o1_loss * mask) / K.sum(mask)
    o2_loss = K.sum(K.binary_crossentropy(o2, po2), 2, keepdims=True)
    o2_loss = K.sum(o2_loss * mask) / K.sum(mask)

    loss = (s1_loss + s2_loss) + (o1_loss + o2_loss)

    train_model.add_loss(loss)
    train_model.compile(optimizer=Adam(learning_rate))
    train_model.summary()

    return subject_model, object_model, train_model

def extract_items(text_in, _t1, _t2, subject_model, object_model):
    """根据输入文本通过模型预测提取三元组"""
    _k1, _k2 = subject_model.predict([_t1, _t2]) 
    _k1, _k2 = np.where(_k1[0] > 0.5)[0], np.where(_k2[0] > 0.4)[0]
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i-1: j] 
            _subjects.append((_subject, i, j)) 

    if _subjects:
        R = []
        _t1 = np.repeat(_t1, len(_subjects), 0)
        _t2 = np.repeat(_t2, len(_subjects), 0)
        _k1, _k2 = np.array([_s[1:] for _s in _subjects]).T.reshape((2, -1, 1))
        _o1, _o2 = object_model.predict([_t1, _t2, _k1, _k2])
        for i,_subject in enumerate(_subjects):
            _oo1, _oo2 = np.where(_o1[i] > 0.5), np.where(_o2[i] > 0.4)
            for _ooo1, _c1 in zip(*_oo1):
                for _ooo2, _c2 in zip(*_oo2):
                    if _ooo1 <= _ooo2 and _c1 == _c2:
                        _object = text_in[_ooo1-1: _ooo2]
                        _predicate = id2predicate[_c1]
                        R.append((_subject[0], _predicate, _object))
                        break
        zhuanji, gequ = [], []
        for s, p, o in R[:]:
            if p == u'妻子':
                R.append((o, u'丈夫', s))
            elif p == u'丈夫':
                R.append((o, u'妻子', s))
            if p == u'所属专辑':
                zhuanji.append(o)
                gequ.append(s)
        spo_list = set()
        for s, p, o in R:
            if p in [u'歌手', u'作词', u'作曲']:
                if s in zhuanji and s not in gequ:
                    continue
            spo_list.add((s, p, o))
        return list(spo_list)
    else:
        return []

class Evaluate(Callback):
    def __init__(self, tokenizer, subject_model, object_model):
        self.F1 = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
        self.tokenizer = tokenizer
        self.subject_model = subject_model
        self.object_model = object_model
        
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights('best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

    def evaluate(self):
        """模型评估"""
        orders = ['subject', 'predicate', 'object']
        A, B, C = 1e-10, 1e-10, 1e-10
        F = open('dev_pred.json', 'w', encoding='utf-8')
        
        for d in tqdm(valid_data,desc='evaluate:'):
            _t1, _t2 = self.tokenizer.encode(first=d['text'])
            _t1, _t2 = np.array([_t1]), np.array([_t2])
            R = set(extract_items(d['text'], _t1, _t2, self.subject_model, self.object_model))
            T = set([tuple(spo) for spo in d['spo_list']])
            A += len(R & T)
            B += len(R)
            C += len(T)
            s = json.dumps({
                'text': d['text'],
                'spo_list': [
                    dict(zip(orders, spo)) for spo in T
                ],
                'spo_list_pred': [
                    dict(zip(orders, spo)) for spo in R
                ],
                'new': [
                    dict(zip(orders, spo)) for spo in R - T
                ],
                'lack': [
                    dict(zip(orders, spo)) for spo in T - R
                ]
            }, ensure_ascii=False, indent=4)
            F.write(s + '\n')
        F.close()
        return 2 * A / (B + C), A / B, A / C


def test(test_data, tokenizer, subject_model, object_model):
    """输出测试结果
    """
    orders = ['subject', 'predicate', 'object', 'object_type', 'subject_type']
    F = open('test_pred.json', 'w', encoding='utf-8')
    for d in tqdm(iter(test_data)):
        R = set(extract_items(d['text'], tokenizer, subject_model, object_model))
        s = json.dumps({
            'text': d['text'],
            'spo_list': [
                dict(zip(orders, spo + ('', ''))) for spo in R
            ]
        }, ensure_ascii=False)
        F.write(s + '\n')
    F.close()

if __name__ == '__main__':
    subject_model, object_model, train_model = create_model()
    train_D = data_generator(train_data, tokenizer, predicate2id, batch_size=4)
    evaluator = Evaluate(tokenizer, subject_model, object_model)

    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=1000,
                              epochs=10,
                              callbacks=[evaluator]
                              )
    test(test_data, tokenizer, subject_model, object_model)
