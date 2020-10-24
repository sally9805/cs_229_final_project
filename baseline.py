#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 3.6

import csv
from sklearn.model_selection import train_test_split
from sklearn import metrics
import naive_bayes
import numpy as np

# Step 1: Read Data
messages = []
labels = []

with open("naive_bayes_data.tsv", 'r', newline='', encoding='utf8') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')

    for message, label in reader:
    	if label != '':
	        messages.append(message)
	        labels.append(1 if label == '1' else 0)

# Step 2: Randomly split into train and test
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Step 3: Use Model
nb_model = naive_bayes.train_naive_bayes_model(X_train, y_train)
predicted = naive_bayes.predict(nb_model, X_test)
print(metrics.confusion_matrix(y_test, predicted, labels=[0,1]))

# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, predicted, labels=[0,1]))