#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import re
from tqdm import tqdm
import time
import sys
from gensim.models.word2vec import Word2Vec

model = Word2Vec.load('word2vec.model')
vocab = dict(float)
f = open('corpus/pku_training.utf8')
for line in f:
    arr = line.split(' ')
    for i in arr:
        vocab[i] += 1

for key, value in vocab:
    print key, value
