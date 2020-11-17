#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 3.6

import csv
import naive_bayes
import numpy as np
from numpy.random import RandomState
import pandas as pd
from autocorrect import Speller
import re

spell = Speller(lang='en')

messages = []
labels = []

# with open("naive_bayes_data.tsv", 'r', newline='', encoding='utf8') as tsv_file:
#     reader = csv.reader(tsv_file, delimiter='\t')

#     for message, label in reader:
#     	if label == '1' or label == '0':
# 	        temp = message.replace('\\n', '')
# 	        messages.append(spell(temp))
# 	        labels.append(1 if label == '1' else 0)

# with open("data.tsv", 'w', encoding='utf8', newline='') as tsv_file:
#         tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
#         for message, label in zip(messages, labels):
#             tsv_writer.writerow([label, message])

# df = pd.read_csv('cleaned_data.tsv', delimiter='\t')
# rng = RandomState()

# train = df.sample(frac=0.8, random_state=rng)
# train.to_csv('train.tsv', index=False, sep = '\t')
# test = df.loc[~df.index.isin(train.index)]
# test.to_csv('test.tsv', index=False, sep = '\t')

with open("train.tsv", 'r', newline='', encoding='utf8') as tsv_file:
	reader = csv.reader(tsv_file, delimiter='\t')
	for label, message in reader:
		if label == '1':
			messages.append(message)
			labels.append(1)
print(len(labels))
with open("train_1.tsv", 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for message, label in zip(messages, labels):
            tsv_writer.writerow([label, message])	