import pandas as pd
import os
import gensim
import csv
import nltk as nl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import naive_bayes
import numpy as np
import xgboost as xgb

messages = []
labels = []
vector_size = 300
with open("eda_train.tsv", 'r', newline='', encoding='utf8') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for label, message in reader:
    	if label != '':
	        messages.append(message)
	        labels.append(1 if label == '1' else 0)
tokens = []
for sentence in messages:
	tokens.append(naive_bayes.get_words(sentence))
model = gensim.models.Word2Vec(tokens, size=vector_size, min_count=1, workers=4, sg=1)

# print(model.wv.vocab)

output = []
for token in tokens: # len(tokens) = 668
	mean = np.zeros(vector_size)
	for word in token:
		# ind = model.wv.vocab[word].index # dict key: 'female', value: self defined object, ind 9
		# mean = np.add(mean, model.wv.vectors[ind]) # list: syn0[9] = female vector
		mean = np.add(mean, model.wv[word])
	length = 1
	if len(token) != 0:
		length = len(token)
	output.append(np.true_divide(mean, length))

# base line: logistic regression
# clf = LogisticRegression(random_state=0, solver='lbfgs').fit(output, labels)
# score = clf.score(output, labels)
# print("logstic regression: " + str(score))

# xgboost
labels = np.asarray(labels)
output = np.asarray(output)
# X_train, X_test, y_train, y_test = train_test_split(output, labels, test_size=0.2, random_state=42)
X_train, y_train = output, labels
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)

X_test = []
y_test = []
with open("test.tsv", 'r', newline='', encoding='utf8') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for label, message in reader:
    	if label != '':
	        X_test.append(message)
	        y_test.append(1 if label == '1' else 0)

test_tokens = []
for sentence in X_test:
	test_tokens.append(naive_bayes.get_words(sentence))
model = gensim.models.Word2Vec(test_tokens, size=vector_size, min_count=1, workers=4, sg=1)

test_output = []
for token in test_tokens: # len(tokens) = 668
	test_mean = np.zeros(vector_size)
	for word in token:
		# ind = model.wv.vocab[word].index # dict key: 'female', value: self defined object, ind 9
		# mean = np.add(mean, model.wv.vectors[ind]) # list: syn0[9] = female vector
		test_mean = np.add(test_mean, model.wv[word])
	length = 1
	if len(token) != 0:
		length = len(token)
	test_output.append(np.true_divide(test_mean, length))

X_test = np.asarray(test_output)
y_test = np.asarray(y_test)

y_pred = xgb_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))