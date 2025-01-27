import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

import pickle

n_splits = 5
output_file = f'model.pkl'

# Data preparation.

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

columns = [
    'Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount',
    'Savings', 'Employment', 'InstallmentRate', 'PersonalStatus', 'OtherDebtors',
    'ResidenceDuration', 'Property', 'Age', 'OtherInstallments', 'Housing',
    'ExistingCredits', 'Job', 'NumberDependents', 'Telephone', 'ForeignWorker', 'Target'
]

df = pd.read_csv(url, sep=' ', header=None, names=columns)

strings = list(df.dtypes[df.dtypes == 'object'].index)
for col in strings:
    df[col] = df[col].str.lower().str.strip().str.replace(' ', '_')

df.columns = (df.columns
              .str.replace(r'(?<!^)(?=[A-Z])', '_', regex=True)
              .str.lower()
              .str.replace(' ', '_')
              .str.strip())

df['target'] = df['target'].apply(lambda x: 0 if x == 1 else 1)

numerical = ['duration', 'credit_amount', 'installment_rate', 'residence_duration', 'age',
             'existing_credits', 'number_dependents']

categorical = ['status', 'credit_history', 'purpose', 'savings',
               'employment', 'personal_status', 'property', 'other_installments',
               'housing', 'job', 'foreign_worker']

df_train, df_test = train_test_split(df, test_size=0.20, random_state=1)
df_train_set, df_val = train_test_split(df_train, test_size=0.20, random_state=1)

# Training.

def train(df_train, y_train):
  dicts = df_train[categorical + numerical].to_dict(orient='records')
  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(dicts)

  model = XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False)

  model.fit(X_train, y_train)

  return dv, model

def predict(df, dv, model):
  dicts = df[categorical + numerical].to_dict(orient='records')

  X = dv.transform(dicts)
  y_pred = model.predict_proba(X)[:, 1]

  return y_pred

# Validation

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_train):
    df_train_set = df_train.iloc[train_idx]
    df_val = df_train.iloc[val_idx]

    y_train = df_train_set.income.values
    y_val = df_val.income.values

    dv, model = train(df_train_set, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1


print('validation results.')
print('XGB. %.3f +- %.3f.' % (np.mean(scores), np.std(scores)))

# Training the final model.

print('training the final model.')

dv, model = train(df_train, df_train.target.values)
y_pred = predict(df_test, dv, model)

y_test = df_test.target.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc. {auc}.')

# Save the model.

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}.')