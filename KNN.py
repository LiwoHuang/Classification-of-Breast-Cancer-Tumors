from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from memory_profiler import memory_usage
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # Read data
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

    # KNN model
    knn = KNeighborsClassifier()

    param_grid = { 
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],  
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  
    }

    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')  

    mem_usage_knn = memory_usage((grid_search.fit, (X_train, y_train)))  

    best_knn = grid_search.best_estimator_
    best_params = grid_search.best_params_
    val_accuracy = np.round(grid_search.best_score_, 4)  
    test_accuracy = np.round(accuracy_score(y_test, best_knn.predict(X_test)), 4)  

    print('Best Estimator:', best_knn)  
    print('Best parameters:', best_params)
    print("Best estimator's validation accuracy:", val_accuracy)
    print("Best estimator's test accuracy:     ", test_accuracy)
    print('Memory usage:  ', np.round(np.mean(mem_usage_knn), 1), 'MiB ±', np.round(np.std(mem_usage_knn), 1))
    print('Training time: ', np.round(np.mean(grid_search.cv_results_['mean_fit_time']), 3),'sec ±',np.round(np.std(grid_search.cv_results_['mean_fit_time']), 3))
    print('Inference time:', np.round(np.mean(grid_search.cv_results_['mean_score_time']), 3),'sec ±',np.round(np.std(grid_search.cv_results_['mean_score_time']), 3))

if __name__ == '__main__':
    main()