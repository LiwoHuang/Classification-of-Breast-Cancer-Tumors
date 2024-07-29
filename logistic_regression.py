import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from memory_profiler import memory_usage

data = []
with open('./wdbc.data', mode ='r') as file:
    csv_data = csv.reader(file)
    for line in csv_data:
        data.append(line)
data = np.array(data)

X = data[:, 2:]
y = (data[:, 1] == 'M').astype(int)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=0)

print(f'X_train.shape: {X_train.shape},  y_train.shape: {y_train.shape},  count(B): {np.sum(y_train == 0)} ({np.round(np.sum(y_train==0)/len(y_train)*100, 1)}%)')
print(f'X_val.shape:   {X_val.shape  },  y_val.shape:   {y_val.shape  },  count(B): {np.sum(y_val == 0) }  ({np.round(np.sum(y_val==0)/len(y_val)*100, 1)}%)')
print(f'X_test.shape:  {X_test.shape },  y_test.shape:  {y_test.shape },  count(B): {np.sum(y_test == 0)}  ({np.round(np.sum(y_test==0)/len(y_test)*100, 1)}%)')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=0)
model.fit(X_train, y_train)

param_grid = {
  'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
  'penalty': ['l2'], 
  'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'] 
}

grid = GridSearchCV(LogisticRegression(multi_class='multinomial', random_state=0), param_grid, cv=5, scoring='accuracy')

mem_usage = memory_usage((grid.fit, (X_train, y_train)))

print('Best Estimator:', grid.best_estimator_)
print('Best parameters:', grid.best_params_)

print('Best estimator\'s validation accuracy:', np.round(grid.best_score_, 4))
print('Best estimator\'s test accuracy:      ', np.round(np.mean(grid.best_estimator_.predict(X_test) == y_test), 4))

print(
    'Training time: ',
    np.round(np.mean(grid.cv_results_['mean_fit_time']), 3),
    'sec ±',
    np.round(np.mean(grid.cv_results_['std_fit_time']), 3),
    '\nInference time:',
    np.round(np.mean(grid.cv_results_['mean_score_time']), 3),
    'sec ±',
    np.round(np.mean(grid.cv_results_['std_score_time']), 3),
    '\nMemory usage:  ',
    np.round(np.mean(mem_usage), 1),
    'MiB ±',
    np.round(np.std(mem_usage), 1),
)