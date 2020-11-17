# from simpletransformers.classification import ClassificationModel
# import pandas as pd
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('naive_bayes_data.tsv', sep='\t', header=0)
# df_bert = pd.DataFrame({
#     'id':range(len(df)),
#     'label':(df['labels'] == 1.0).astype(int),
#     'alpha':['a']*df.shape[0],
#     'text': df['text'].replace(r'\n', ' ', regex=True)
# })
# train, dev = train_test_split(df_bert, test_size=0.2)

# # save data files
# # train.to_csv('train_bert.tsv', sep='\t', index=False, header=False)
# # dev.to_csv('dev_bert.tsv', sep='\t', index=False, header=False)


# # Create a TransformerModel
# model = ClassificationModel('roberta', 'roberta-base', use_cuda=False)

# # Train the model
# model.train_model(train)

# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(dev)
# print(result, model_outputs, wrong_predictions)

# from simpletransformers.classification import ClassificationModel
# import pandas as pd
# import logging


# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# # Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
# train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
# train_df = pd.DataFrame(train_data)

# eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
# eval_df = pd.DataFrame(eval_data)

# # Create a ClassificationModel
# model = ClassificationModel('roberta', 'roberta-base', use_cuda=False) # You can set class weights by using the optional weight argument

# # Train the model
# model.train_model(train_df)

# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
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
with open("naive_bayes_data.tsv", 'r', newline='', encoding='utf8') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for message, label in reader:
    	if label != '':
	        messages.append(message)
	        labels.append(1 if label == '1' else 0)
tokens = []
for sentence in messages:
	tokens.append(naive_bayes.get_words(sentence))