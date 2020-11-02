from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('naive_bayes_data.tsv', sep='\t', header=0)
df_bert = pd.DataFrame({
    'id':range(len(df)),
    'label':(df['labels'] == 1.0).astype(int),
    'alpha':['a']*df.shape[0],
    'text': df['text'].replace(r'\n', ' ', regex=True)
})
train, dev = train_test_split(df_bert, test_size=0.2)

# save data files
# train.to_csv('train_bert.tsv', sep='\t', index=False, header=False)
# dev.to_csv('dev_bert.tsv', sep='\t', index=False, header=False)


# Create a TransformerModel
model = ClassificationModel('roberta', 'roberta-base', use_cuda=False)

# Train the model
model.train_model(train)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(dev)
print(result, model_outputs, wrong_predictions)