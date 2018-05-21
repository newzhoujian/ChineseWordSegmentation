#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
import gensim
import re
from tqdm import tqdm
from keras.models import load_model
import time
import sys
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model

with open('corpus/pku_training_processed.utf8', 'rb') as f:
    texts = f.read().decode('utf8')

print texts[0:300]
sentences = texts.split('\n')
print sentences[0]

'''
def clean(s):
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

texts = u''.join(map(clean, sentences)) # 把所有的词拼接起来
'''

texts = u''.join(sentences) # 把所有的词拼接起来

print 'Length of texts is %d' % len(texts)
print 'Example of texts: \n', texts[0:300]

sentences = re.split(u'[，。！？、‘’“”（）]', texts)
print 'Sentences number:', len(sentences)
print 'Sentence Example:\n', sentences[0]

data = [] #生成训练样本
label = []
def get_xy(sentences):
    s = re.findall('(.)/(.)', sentences)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])

for i in sentences:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

chars = [] #统计所有字，跟每个字编号
for i in data:
    chars.extend(i)
chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars)+1)



maxlen = 32

model1 = load_model('mymodel.h5')

zy = {'BE': 0.853,
      'BM': 0.147,
      'EB': 0.487,
      'ES': 0.513,
      'ME': 0.654,
      'MM': 0.346,
      'SB': 0.573,
      'SS': 0.427
     }

zy = {i:np.log(zy[i]) for i in zy.keys()}


def viterbi(nodes):
    paths = {'B':nodes[0]['B'], 'S':nodes[0]['S']}
    for l in range(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]

def simple_cut(s):
    if s:
        r = model1.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]),
                           verbose=True)[0][:len(s)]
        print r
        r = np.log(r)
        print r
        nodes = [dict(zip(['S', 'B', 'M', 'E'], i[1:])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['S', 'B']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []

not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')
def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result


sen = u'王思斌指出，当前，中美关系发展处在一个重要阶段。我高度重视发展两国关系，珍视同总统先生的良好工作关系。' \
      u'希望双方认真落实我同总统先生在北京会晤时达成的共识，保持高层及各级别交往，相互尊重、互利互惠，聚焦合作、管控分歧，' \
      u'推动两国关系健康稳定向前发展。经贸合作一直是中美关系的压舱石和推进器。上周，中美双方在北京就经贸问题进行了坦诚、高效、建设性的磋商。' \
      u'双方团队可以保持沟通，争取找到妥善解决存在问题的办法，取得互利双赢的成果。'
# sen = u'结婚的和尚未结婚的'
# sen = u'乒乓球拍卖完了'
print cut_word(sen)