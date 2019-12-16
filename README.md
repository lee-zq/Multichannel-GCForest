## A implementation of gcForest (or named deep Forest)
### Introduction
GcForest paper url： ( https://arxiv.org/abs/1702.08835 ).
The official release of the gcForest code used in paper at [here](https://github.com/kingfengji/gcforest). 

This project implements a **multi-channel deep-forest** based on the pylablanche's [work](https://github.com/pylablanche/gcForest). Thus, the project allows you to input multi-channels images.
### Using GCForest
The project contains a small sample data set, so you can directly run *main.py* for training, and you can directly see the test results, as follows:
```
cd ./   
python main.py  
```
result:
```
Slicing Images...
sliced_imgs shape after MGS：: (2700, 972)
Training MGS Random Forests...
Adding/Training Layer, n_layer=1
Layer validation accuracy = 0.9166666666666666
Adding/Training Layer, n_layer=2
Layer validation accuracy = 0.9333333333333333
Adding/Training Layer, n_layer=3
Layer validation accuracy = 0.9333333333333333
模型训练时间： 5.067959200000001
Slicing Images...
sliced_imgs shape after MGS：: (2700, 972)
测试集结果：
测试运行时间 1.4750 s
accuracy: 0.88
kappa: 0.76
              precision    recall  f1-score   support

           0       0.84      0.94      0.89       150
           1       0.93      0.82      0.87       150

    accuracy                           0.88       300
   macro avg       0.89      0.88      0.88       300
weighted avg       0.89      0.88      0.88       300

[[141   9]
 [ 27 123]]
Confusion matrix, without normalization
```
