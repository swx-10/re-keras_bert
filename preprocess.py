import os, re
import json
import numpy as np
from tqdm import tqdm

# 训练文件
train_file = os.path.join(os.path.dirname(__file__),'original_datasets','train_data.json')
c_train_file = os.path.join(os.path.dirname(__file__),'datasets','train_data_me.json')
# 开发文件
dev_file = os.path.join(os.path.dirname(__file__),'original_datasets','dev_data.json')
c_dev_file = os.path.join(os.path.dirname(__file__),'datasets','dev_data_me.json')
# 类别文件
schema_file = os.path.join(os.path.dirname(__file__),'original_datasets', 'all_50_schemas.json')
c_schema_file = os.path.join(os.path.dirname(__file__),'datasets', 'all_50_schemas_me.json')

def convert_datafile(json_file, json_convert_file):
    """转换json语料"""
    json_data = []
    for line in tqdm(open(json_file, encoding='utf-8').readlines()):
        json_item = json.loads(line)
        text = json_item['text']
        spo_list = []
        _spo_list = json_item['spo_list']
        for _spo in _spo_list:
            spo = (_spo['subject'], _spo['predicate'], _spo['object'])
            spo_list.append(spo)
        json_data.append({'text':text, 'spo_list':spo_list})
    with open(json_convert_file,'w') as f:
        json.dump(json_data,f)

def convert_schemafile(schema_file, convert_file):
    """转换schema格式文件"""
    id2predicate,predicate2id = {},{}
    lines = [line for line in open(schema_file, encoding='utf-8').readlines()]
    predicates = set() 
    for i,line in enumerate(lines):
        predicate = json.loads(line)['predicate']
        predicates.add(predicate)
    
    id2predicate = { i:p for i,p in enumerate(predicates)}
    predicate2id = { p:i for i,p in enumerate(predicates)}
    with open(convert_file,'w') as f:
        json.dump((id2predicate,predicate2id),f)

def split_dataset(dataset, split_rate=0.8):
    data_size = len(dataset)
    random_order = list(range(data_size))
    np.random.shuffle(random_order)

    data1 = [dataset[j] for i, j in enumerate(random_order) if i <= data_size * split_rate]
    data2 = [dataset[j] for i, j in enumerate(random_order) if i > data_size * split_rate]
    return [data1, data2]

def repair(d):
        d['text'] = d['text'].lower()
        something = re.findall(u'《([^《》]*?)》', d['text'])
        something = [s.strip() for s in something]
        zhuanji = []
        gequ = []
        for sp in d['spo_list']:
            sp = list(sp)
            sp[0] = sp[0].strip(u'《》').strip().lower()
            sp[2] = sp[2].strip(u'《》').strip().lower()
            for some in something:
                if sp[0] in some and d['text'].count(sp[0]) == 1:
                    sp[0] = some
            if sp[1] == u'所属专辑':
                zhuanji.append(sp[2])
                gequ.append(sp[0])
        spo_list = []
        for sp in d['spo_list']:
            if sp[1] in [u'歌手', u'作词', u'作曲']:
                if sp[0] in zhuanji and sp[0] not in gequ:
                    continue
            spo_list.append(tuple(sp))
        d['spo_list'] = spo_list

if __name__=='__main__':
    convert_datafile(train_file, c_train_file)
    convert_datafile(dev_file, c_dev_file)
    convert_schemafile(schema_file,c_schema_file)


