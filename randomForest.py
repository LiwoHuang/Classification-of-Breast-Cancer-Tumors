import csv
import numpy as np
from sklearn.model_selection import train_test_split
from memory_profiler import memory_usage
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


data = []
with open('wdbc.data', mode ='r') as file:
    csv_data = csv.reader(file)
    for line in csv_data:
        data.append(line)
data = np.array(data)

X = data[:, 2:]
y = (data[:, 1] == 'M').astype(int)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=0)

# print(f'X_train.shape: {X_train.shape},  y_train.shape: {y_train.shape},  count(B): {np.sum(y_train == 0)} ({np.round(np.sum(y_train==0)/len(y_train)*100, 1)}%)')
# print(f'X_val.shape:   {X_val.shape  },  y_val.shape:   {y_val.shape  },  count(B): {np.sum(y_val == 0) }  ({np.round(np.sum(y_val==0)/len(y_val)*100, 1)}%)')
# print(f'X_test.shape:  {X_test.shape },  y_test.shape:  {y_test.shape },  count(B): {np.sum(y_test == 0)}  ({np.round(np.sum(y_test==0)/len(y_test)*100, 1)}%)')

rfc = RandomForestClassifier(max_depth = 2, random_state = 0)
rfc.fit(X_train, y_train)

validationScore = rfc.score(X_val, y_val)
print("Validation accuracy: ", validationScore)

testScore = rfc.score(X_test, y_test)
print("Test accuracy: ", testScore)
