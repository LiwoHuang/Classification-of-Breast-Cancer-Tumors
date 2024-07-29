import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from memory_profiler import memory_usage

# read data and preprocessing
def load_and_preprocess_data():
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
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Model
def setup_and_run_svm(X_train, y_train):
    svm_model = SVC(random_state=0)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        scoring='accuracy',
        refit='accuracy',
        cv=5,
        verbose=3,
    )

    mem_usage = memory_usage((grid_search.fit, (X_train, y_train)))
    return grid_search, mem_usage

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    grid_search, mem_usage = setup_and_run_svm(X_train, y_train)

    print('Best Estimator:', grid_search.best_estimator_)
    print('Best parameters:', grid_search.best_params_)
    print('Best estimator\'s validation accuracy:', np.round(grid_search.best_score_, 4))
    print('Best estimator\'s test accuracy:', np.round(np.mean(grid_search.best_estimator_.predict(X_test) == y_test), 4))
    print(
        'Training time: ',
        np.round(np.mean(grid_search.cv_results_['mean_fit_time']), 3),
        'sec ±',
        np.round(np.mean(grid_search.cv_results_['std_fit_time']), 3),
        '\nInference time:',
        np.round(np.mean(grid_search.cv_results_['mean_score_time']), 3),
        'sec ±',
        np.round(np.mean(grid_search.cv_results_['std_score_time']), 3),
        '\nMemory usage:  ',
        np.round(np.mean(mem_usage), 1),
        'MiB ±',
        np.round(np.std(mem_usage), 1),
    )
