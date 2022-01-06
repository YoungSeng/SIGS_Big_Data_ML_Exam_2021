# SIGS Big Data ML Exam 2021

2021 September.27 First set up the Github

Team Members：

庄昊霖 2021214563
杨思程 2021214415
蔡紫宴 2021214571
李御智 2021214570
常鑫	2021214589

---------------------------------------------------------------
## submission 1：Tue Sep 28 2021 20:15:31

resnet34 模型 迭代2次

Public Score：0.04047
rank：2/3

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle.png)

## submission 2：Thu Sep 30 2021 15:43:51

resnet34 模型 迭代300次
Best Prec@1: 26.865%

Public Score：0.46225
rank：2/3

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle-0930.png)

## submission 3：Thu Jan 06 2022 22:02:43

<details>
<summary>250次迭代结果</summary>

```python
--------------------epoch: 249--------------------
Training start ...
100%|█████████████████████████████████████████| 138/138 [00:37<00:00,  3.70it/s]
Wait for validation ...
* Acc@1 27.756% Acc@5 36.205%.
* Best Prec@1: 27.987%
Finish !

Process finished with exit code 0
```
</details>

resnet50 模型 迭代250次 Best Prec@1: 27.987%

Public Score：0.47945 rank：4/8

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle-0106.jpg)

## submission 4


## 改进方向：Jan 06 2022

### 数据清洗与数据增强

发现训练集总共有17555组数据，但是每种类别的数量逐次递减，从0对应的800一直到100对应的8，所以存在不平衡数据：


<details>
<summary>每种类别的数量</summary>

```python
[800, 763, 729, 696, 665, 635, 606, 579, 553, 528, 504, 482, 460, 439, 419, 400, 382, 365, 349, 333, 318, 304, 290, 277, 264, 252, 241, 230, 220, 210, 200, 191, 183, 175, 167, 159, 152, 145, 139, 132, 126, 121, 115, 110, 105, 100, 96, 91, 87, 83, 80, 76, 72, 69, 66, 63, 60, 57, 55, 52, 50, 48, 46, 43, 41, 40, 38, 36, 34, 33, 31, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 8]
```
</details>

### 尝试不同模型

尽管原Food-101数据集每类样本有750 training and 250 test images，现在我们只有17555组数据。
但是在 [paperswithcode](https://paperswithcode.com/dataset/food-101) 上的Benchmarks方法值得我们借鉴，同时一些SOTA方法有开源代码也值得我们学习：
1. 细粒度的图像分类——不需要Extra Training Data的方法：
    1. EfficientNet-B7
    2. Assemble-ResNet-FGVC-50
    3. NAT-M1/M2/M3/M4(神经结构搜索)
2. 图像分类——不需要Extra Training Data的方法：
   1. TWIST(ResNet-50)
   2. NNCLR
3. 图像压缩——没有Accuracy
4. 多模态文本和图像分类——Bert + InceptionV3，使用预训练模型BERT，不符合kaggle比赛要求
5. 文档文本分类——using BERT and CNNs，使用预训练模型BERT，不符合kaggle比赛要求

其他不是SOTA但也很经典的模型也值得尝试：
1. ResNet-34
2. NASNet-A
3. Inception-v2
4. ResNet-50
5. DenseNet-201
6. ResNet-152
7. Xception
8. Inception-ResNet-v2
9. ResNeXt-101
10. NASNet-A
11. AmoebaNet-A
12. SENet
13. AmoebaNet-C
14. EfficientNet-B7
