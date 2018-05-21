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


def get_Xy(sentence):
    """将 sentence 处理成 [word1, w2, ..wn], [tag1, t2, ...tn]"""
    words_tags = re.findall('(.)/(.)', sentence)
    if words_tags:
        words_tags = np.asarray(words_tags)
        words = words_tags[:, 0]
        tags = words_tags[:, 1]
        return words, tags  # 所有的字和tag分别存为 data / label
    return None

datas = list()
labels = list()
print 'Start creating words and tags data ...'
for sentence in tqdm(iter(sentences)):
    result = get_Xy(sentence)
    if result:
        datas.append(result[0])
        labels.append(result[1])

print 'Length of datas is %d' % len(datas)
print 'Example of datas: ', datas[0]
print 'Example of labels:', labels[0]

df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
#　句子长度
df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
print df_data.head(2)


'''
import matplotlib.pyplot as plt
df_data['sentence_len'].hist(bins=100)
plt.xlim(0, 100)
plt.xlabel('sentence_length')
plt.ylabel('sentence_num')
plt.title('Distribution of the Length of Sentence')
plt.show()
'''



# 1.用 chain(*lists) 函数把多个list拼接起来
from itertools import chain
all_words = list(chain(*df_data['words'].values))
# 2.统计所有 word
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1)  # 注意从1开始，因为我们准备把0作为填充值
tags = ['X', 'S', 'B', 'M', 'E']
tag_ids = range(len(tags))

# 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)

vocab_size = len(set_words)
print 'vocab_size={}'.format(vocab_size)

model = Word2Vec.load('word2vec.model')
vocab = defaultdict(float)
f = open('corpus/pku_training.utf8')
for line in f:
    arr = line.split(' ')
    for i in arr:
        i.decode('utf8')
        vocab[i] += 1

total_inside_new_embed = 0
miss = 0
word_vecs = {}

for pair in vocab:
    word = gensim.utils.to_unicode(pair)
    # word = pair
    if word in model:
        total_inside_new_embed += 1
        word_vecs[pair] = np.array([w for w in model[word]])
    else:
        miss = miss + 1
        word_vecs[pair] = np.array([0.] * model.vector_size)
#sys.exit()

max_len = 32
def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids


'''
def X_padding(words):
    """把 words 转为 embedding 形式，并自动补全位 max_len 长度。"""
    ids = []
    for pair in words:
        word = gensim.utils.to_unicode(pair)
        if word in model:
            ids.append(np.array([w for w in model[word]]))
        else:
            ids.append(np.array([0.] * model.vector_size))
    # ids = list(vocab[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([np.array([0.] * model.vector_size)]*(max_len-len(ids)))  # 短则补全
    return ids
def y_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    ids = list(tag2id[tags])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids)))  # 短则补全
    return ids
'''

mp = dict()
mp['X'] = np.array([1, 0, 0, 0, 0])
mp['S'] = np.array([0, 1, 0, 0, 0])
mp['B'] = np.array([0, 0, 1, 0, 0])
mp['M'] = np.array([0, 0, 0, 1, 0])
mp['E'] = np.array([0, 0, 0, 0, 1])


def y_padding(tags):
    ids = []
    for _ in tags:
        ids.append(mp[_])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([mp['X']] * (max_len - len(ids)))  # 短则补全
    return ids

df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
'''
max_len = 32
def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = list(word2id[words])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

def y_padding(tags):
    """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
    ids = list(tag2id[tags])
    if len(ids) >= max_len:  # 长则弃掉
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) # 短则补全
    return ids

df_data['X'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)

print df_data['X']
print df_data['y']
'''

X = np.asarray(list(df_data['X'].values))
y = np.asarray(list(df_data['y'].values))
print 'X.shape={}, y.shape={}'.format(X.shape, y.shape)
print 'Example of words: ', df_data['words'].values[0]
print 'Example of X: ', X[0]
print 'Example of tags: ', df_data['tags'].values[0]
print 'Example of y: ', y[0]

print X.shape
print y.shape
sys.exit()

# model
word_size = model.vector_size
maxlen = max_len
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model

sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(len(chars)+1, word_size, input_length=maxlen, mask_zero=True)(sequence)
blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
model1 = Model(input=sequence, output=output)
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 1024
history = model1.fit(X, y, batch_size=batch_size, epochs=50)
print 'DONE!'
print 'SAVING MODEL...'
model1.save('mymodel.h5')
#  keras.models.load_model(filepath)
print 'OK, model saved!'

zy = {'BE': 0.5,
      'BM': 0.5,
      'EB': 0.5,
      'ES': 0.5,
      'ME': 0.5,
      'MM': 0.5,
      'SB': 0.5,
      'SS': 0.5
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
        r = model1.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['S', 'B','M','E'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
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

