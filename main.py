from GCForest import gcForest
import gzip
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics as metrics
import numpy as np
import cv2
import os, random
import time


# 加载数据集
def get_data(data_path,p_dir,p_count,n_dir,n_count): # w是训练集图片边长
    data_numpy = []
    labels = []
    p_list = os.listdir(os.path.join(data_path, p_dir))
    n_list = os.listdir(os.path.join(data_path, n_dir))
    p_list = random.sample(p_list, p_count)
    n_list = random.sample(n_list, n_count)
    for i in p_list:
        img = cv2.imread(os.path.join(data_path, p_dir, i))
        data_numpy.append(img)
        labels.append(1)
    for j in n_list:
        img = cv2.imread(os.path.join(data_path, n_dir, j))
        data_numpy.append(img)
        labels.append(0)
    data_numpy = np.array(data_numpy)
    labels = np.array(labels)
    shuffle_ix = np.random.permutation(np.arange(len(labels)))
    data = data_numpy[shuffle_ix]
    labels = labels[shuffle_ix]
    return data, labels

# 混淆矩阵评估可视化
def plot_confusion_matrix(cm, classes, normalize=True,
                          title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation= 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],horizontalalignment='center',
                color= "white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel("Predicted labels")
    plt.show()

# 训练过程
def gcf(X_train, X_test,y_train, y_test, cnames):

    clf = gcForest(shape_1X=[20,20,3], n_mgsRFtree=80, window=[18], stride=1,
                 cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101,
                 min_samples_mgs=0.1, min_samples_cascade=0.05, tolerance=0.0, n_jobs=3)

    train_start = time.clock()
    clf.fit(X_train, y_train)  # 模型训练
    train_end = time.clock()
    print('模型训练时间：', train_end-train_start)

    y_pred = clf.predict(X_test)  # 模型测试
    pre_end = time.clock()
    print('测试集结果：')
    print('测试运行时间 %.4f s' % (pre_end-train_end))
    print("accuracy:", metrics.accuracy_score(y_test,y_pred))
    print("kappa:", metrics.cohen_kappa_score(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred,target_names=cnames))

    cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix, classes=cnames,normalize=False,
                          title="Normalized confusion matrix")

if __name__ == '__main__':
    trainset_path = r'Dataset\train'
    testset_path = r'Dataset\test'

    X_tr, y_tr = get_data(trainset_path,'p_samples',150, 'n_samples',150)
    X_te, y_te = get_data(testset_path,'p_samples', 100, 'n_samples',100)
    cnames = ['0', '1']

    X_tr = X_tr/255.0
    X_te = X_te/255.0

    gcf(X_tr, X_te, y_tr, y_te, cnames)
