from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from memory_profiler import memory_usage
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # 读取数据
    data = []
    with open('wdbc.data', mode='r') as file:
        csv_data = csv.reader(file)
        for line in csv_data:
            data.append(line)
    data = np.array(data)

    # 数据处理
    X = data[:, 2:].astype(np.float64)  # 特征提取，从第三列开始
    y = (data[:, 1] == 'M').astype(int)  # M 为恶性 写为1， B 为阴性 写为0

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)  #  stratify=y: 这个参数确保分割后的训练集和测试集中各类别（恶性和良性）的样本比例与原始数据集中的比例相同。

    # 特征标准化。标准化是指将所有特征缩放到同一尺度，提高算法的收敛速度和模型的性能。
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # KNN模型
    knn = KNeighborsClassifier()

    # 网格搜索超参数
    param_grid = {  # 这是一个字典，定义了要尝试的参数组合。
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],  # 'uniform'：所有邻居的权重相同，每个邻居对分类结果的贡献相等。'distance'：权重与距离成反比，即距离越近的邻居对分类结果的影响越大。这种方法可以帮助模型更加重视最近的邻居，可能在某些情况下提高准确率。
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']  # 这个参数指定了计算最近邻的方法。提供了四种选择
    }

    # 设置 GridSearchCV
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')  # 用于对指定的K最近邻（KNN）模型进行系统的超参数搜索和优化，它旨在找到最佳的参数配置，以提高模型的准确性。

    # 记录内存使用情况
    mem_usage_knn = memory_usage((grid_search.fit, (X_train, y_train)))  # fit 方法负责在给定的参数网格上训练模型

    # 结果
    best_knn = grid_search.best_estimator_
    best_params = grid_search.best_params_
    val_accuracy = np.round(grid_search.best_score_, 4)  # 在交叉验证过程中获得的最佳平均准确率
    test_accuracy = np.round(accuracy_score(y_test, best_knn.predict(X_test)), 4)  # 使用在 GridSearchCV 中找到的最佳模型（即 best_knn）对测试集 X_test 进行预测。 accuracy_score(y_test, ...) 计算预测标签与实际标签 y_test 之间的准确率。

    # 输出
    print('Best Estimator:', best_knn)  # 目前只有一个knn
    print('Best parameters:', best_params)
    print("Best estimator's validation accuracy:", val_accuracy)
    print("Best estimator's test accuracy:     ", test_accuracy)
    print('Memory usage:  ', np.round(np.mean(mem_usage_knn), 1), 'MiB ±', np.round(np.std(mem_usage_knn), 1))
    print('Training time: ', np.round(np.mean(grid_search.cv_results_['mean_fit_time']), 3),'sec ±',np.round(np.std(grid_search.cv_results_['mean_fit_time']), 3))
    print('Inference time:', np.round(np.mean(grid_search.cv_results_['mean_score_time']), 3),'sec ±',np.round(np.std(grid_search.cv_results_['mean_score_time']), 3))

if __name__ == '__main__':
    main()