import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from memory_profiler import memory_usage

data = []
with open('wdbc.data', mode='r') as file:
    csv_data = csv.reader(file)
    for line in csv_data:
        data.append(line)
data = np.array(data)

X = data[:, 2:].astype(np.float64)
y = (data[:, 1] == 'M').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

estimator = LogisticRegression(random_state=0, max_iter=10000)
search_space = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['lbfgs', 'saga', 'liblinear'],
    'C': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.5, 0.9]  # Only used if penalty is elasticnet
}

# Filtering out incompatible combinations
if 'liblinear' in search_space['solver']:
    search_space['penalty'] = ['l1', 'l2']  # 'elasticnet' and 'none' are not supported by 'liblinear'

GS = GridSearchCV(
    estimator=estimator,
    param_grid=search_space,
    scoring='accuracy',
    refit='accuracy',
    cv=5,
    verbose=3,
)

if __name__ == '__main__':
    mem_usage = memory_usage((GS.fit, (X_train, y_train)))

    print('Best Estimator:', GS.best_estimator_)
    print('Best parameters:', GS.best_params_)

    print('Best estimator\'s validation accuracy:', np.round(GS.best_score_, 4))
    print('Best estimator\'s test accuracy:      ', np.round(np.mean(GS.best_estimator_.predict(X_test) == y_test), 4))

    print(
        'Training time: ',
        np.round(np.mean(GS.cv_results_['mean_fit_time']), 3),
        'sec ±',
        np.round(np.mean(GS.cv_results_['std_fit_time']), 3),
        '\nInference time:',
        np.round(np.mean(GS.cv_results_['mean_score_time']), 3),
        'sec ±',
        np.round(np.mean(GS.cv_results_['std_score_time']), 3),
        '\nMemory usage:  ',
        np.round(np.mean(mem_usage), 1),
        'MiB ±',
        np.round(np.std(mem_usage), 1),
    )

