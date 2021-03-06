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

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle.png)

## submission 2：Thu Sep 30 2021 15:43:51

resnet34 模型 迭代300次
Best Prec@1: 26.865%

Public Score：0.46225
rank：2/3

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0930.png)

## submission 3：Thu Jan 06 2022 22:02:43

resnet50 模型 迭代250次 Best Prec@1: 27.987%

<details>
<summary>250次迭代结果</summary>

```
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

Public Score：0.47945 rank：4/8

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0106.jpg)

## submission 4-7： Jan 07 2022

ResNet101  batch_size = 196 

<details>
<summary>300次迭代结果</summary>

```
--------------------epoch: 299--------------------
Training start ...
100%|█████████████████████████████████████████| 138/138 [01:02<00:00,  2.22it/s]
Wait for validation ...
* Acc@1 26.139% Acc@5 35.611%.
* Best Prec@1: 26.898%
Finish !
```
</details>

Public Score：0.45594

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0107-1.jpg)

ResNet152  batch_size = 128

<details>
<summary>300次迭代结果</summary>

```
--------------------epoch: 299--------------------
Training start ...
100%|███████████████████████████████████████████| 90/90 [00:50<00:00,  1.79it/s]
Wait for validation ...
* Acc@1 26.502% Acc@5 35.215%.
* Best Prec@1: 26.964%
Finish !
```
</details>

Public Score：0.46522

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0107-2.jpg)

ResNet101  batch_size = 128 Adam

<details>
<summary>300次迭代结果</summary>

```
--------------------epoch: 299--------------------
Training start ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [00:48<00:00,  2.83it/s]
Wait for validation ...
* Acc@1 26.733% Acc@5 35.149%.
* Best Prec@1: 27.129%
Finish !
```
</details>

Public Score：0.45643

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0107-3.jpg)

ResNet152 batch_size = 128 Adam

<details>
<summary>300次迭代结果</summary>

```
--------------------epoch: 299--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [01:04<00:00,  2.13it/s]
Wait for validation ...
* Acc@1 26.403% Acc@5 34.719%.
* Best Prec@1: 26.964%
Finish !
```
</details>

Public Score：0.45358

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0107-4.jpg)

结论：
1. Resnet101和Resnet152对结果没有改善，可能因为数据集比较小；
2. Adam较SGD相比对结果没有改善。

## 改进方向：Jan 06 2022

### 数据清洗与数据增强

发现训练集总共有17555组数据，但是每种类别的数量逐次递减，从0对应的800一直到100对应的8，所以存在不平衡数据：


<details>
<summary>训练集每种类别的数量</summary>

```python
[800, 763, 729, 696, 665, 635, 606, 579, 553, 528, 504, 482, 460, 439, 419, 400, 382, 365, 349, 333, 318, 304, 290, 277, 264, 252, 241, 230, 220, 210, 200, 191, 183, 175, 167, 159, 152, 145, 139, 132, 126, 121, 115, 110, 105, 100, 96, 91, 87, 83, 80, 76, 72, 69, 66, 63, 60, 57, 55, 52, 50, 48, 46, 43, 41, 40, 38, 36, 34, 33, 31, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 8]
```
</details>

### submission 8-10： Jan 07 2022

训练集和验证集整体划分数据集：Jan 07 2022

将原数据集的train和val混合，重新划分80%为训练集，20%为验证集

由于random split划分数据集的随机性，基于Resnet50进行两次实验，基于Resnet101进行一次实验。

Resnet50 第一次实验 

Best Prec@1: 61.865%

<details>
<summary>300次迭代结果</summary>

```
--------------------epoch: 299--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 129/129 [00:33<00:00,  3.90it/s]
Wait for validation ...
* Acc@1 61.355% Acc@5 79.888%.
* Best Prec@1: 61.865%
Finish !
```
</details>

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-1.jpg)

Resnet50 第二次实验

Best Prec@1: 63.201%

<details>
<summary>300次迭代结果</summary>

```
--------------------epoch: 299--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 129/129 [00:54<00:00,  2.37it/s]
Wait for validation ...
* Acc@1 61.671% Acc@5 80.763%.
* Best Prec@1: 63.201%
Finish !
```
</details>

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-2.jpg)

Resnet101 实验

Best Prec@1: 61.404%

<details>
<summary>300次迭代结果</summary>

```
--------------------epoch: 299--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 129/129 [00:41<00:00,  3.12it/s]
Wait for validation ...
* Acc@1 58.076% Acc@5 78.237%.
* Best Prec@1: 61.404%
Finish !
```
</details>

<img width="600" src="https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle-0108-3.jpg"/>

[comment]: <> (![image]&#40;https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle-0108-3.jpg&#41;)

现在出现了一个很奇怪的现象，之前的模型最好的验证集准确率不到30%，而现在验证集准确率接近60%，
但是效果反而变差了，之前在kaggle上的测试集效果比现在好，很奇怪。

验证集共有3030张图片，如果拿1/3(1010)张图片加入训练集，剩下2020张图片作为验证集呢？

### submission 11-14： Jan 08 2022

将验证集的一部分拿来加入训练集，结果如下：

<div class="center">

| train from valid | Best Prec | Public Score |
| :-------------: | :-------------: | :-------------: |
| 1/5 | 28.630% | 0.40272 |
| 1/4 | 28.993% | 0.36943 |
| 1/3 | 24.604% | 0.34950 |
| 1/2 | 30.693% | 0.36608 |

</div>


<details>
<summary>1/5：200次迭代结果</summary>

```
--------------------epoch: 199--------------------
Training start ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 142/142 [00:58<00:00,  2.41it/s]
Wait for validation ...
* Acc@1 28.094% Acc@5 46.535%.
* Best Prec@1: 28.630%
Finish !
```
</details>
<details>
<summary>1/4：200次迭代结果</summary>

```
--------------------epoch: 199--------------------
Training start ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 144/144 [00:35<00:00,  4.07it/s]
Wait for validation ...
* Acc@1 28.289% Acc@5 45.799%.
* Best Prec@1: 28.993%
Finish !
```
</details>
<details>
<summary>1/3：200次迭代结果</summary>

```
--------------------epoch: 199--------------------
Training start ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 146/146 [01:06<00:00,  2.21it/s]
Wait for validation ...
* Acc@1 23.762% Acc@5 40.149%.
* Best Prec@1: 24.604%
Finish !
```
</details>
<details>
<summary>1/2：200次迭代结果</summary>

```
--------------------epoch: 199--------------------
Training start ...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 149/149 [00:39<00:00,  3.81it/s]
Wait for validation ...
* Acc@1 29.637% Acc@5 56.832%.
* Best Prec@1: 30.693%
Finish !
```
</details>


<center><figure><img src="https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-4.jpg" />···<img src="https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-5.jpg" /></figure></center>




<center><figure><img src="https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-6.jpg" />···<img src="https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-7.jpg" /></figure></center>




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


## submission 15-17： Jan 08 2022

densenet121

<details>
<summary>densenet121：200次迭代结果</summary>

```
--------------------epoch: 199--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [00:53<00:00,  2.56it/s]
Wait for validation ...
* Acc@1 25.380% Acc@5 36.007%.
* Best Prec@1: 25.578%
Finish !
```
</details>

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-8.jpg)

densenet161

<details>
<summary>densenet161：200次迭代结果</summary>

```
--------------------epoch: 199--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [01:13<00:00,  1.87it/s]
Wait for validation ...
* Acc@1 25.446% Acc@5 35.644%.
* Best Prec@1: 25.809%
Finish !
```
</details>

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-9.jpg)

efficientnet_b0

<details>
<summary>efficientnet_b0：200次迭代结果</summary>

```
--------------------epoch: 199--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [00:56<00:00,  2.46it/s]
Wait for validation ...
* Acc@1 24.488% Acc@5 35.908%.
* Best Prec@1: 25.149%
Finish !
```
</details>

![image](https://github.com/YoungSeng/SIGS_Big_Data_ML_Exam_2021/blob/master/kaggle_result/kaggle-0108-10.jpg)

## submission 18...： Jan 10 2022

efficientnet_b1

<details>
<summary>efficientnet_b1：250次迭代结果</summary>

```
--------------------epoch: 249--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [00:54<00:00,  2.52it/s]
Wait for validation ...
* Acc@1 25.413% Acc@5 35.611%.
* Best Prec@1: 25.809%
Finish !
```
</details>

efficientnet_b2

<details>
<summary>efficientnet_b2：250次迭代结果</summary>

```
--------------------epoch: 249--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [00:53<00:00,  2.59it/s]
Wait for validation ...
* Acc@1 25.380% Acc@5 35.116%.
* Best Prec@1: 25.611%
Finish !
```
</details>

efficientnet_b3

<details>
<summary>efficientnet_b3：250次迭代结果</summary>

```
--------------------epoch: 249--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [00:53<00:00,  2.56it/s]
Wait for validation ...
* Acc@1 25.380% Acc@5 35.116%.
* Best Prec@1: 26.007%
Finish !
```
</details>

regnet_x_800mf

<details>
<summary>regnet_x_800mf：250次迭代结果</summary>

```
--------------------epoch: 249--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [00:56<00:00,  2.44it/s]
Wait for validation ...
* Acc@1 23.729% Acc@5 34.917%.
* Best Prec@1: 24.257%
Finish !
```
</details>

regnet_x_1_6gf

<details>
<summary>regnet_x_1_6gf：250次迭代结果</summary>

```
--------------------epoch: 249--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [01:31<00:00,  1.51it/s]
Wait for validation ...
* Acc@1 25.380% Acc@5 35.809%.
* Best Prec@1: 25.644%
Finish !
```
</details>

regnet_x_3_2gf

<details>
<summary>regnet_x_3_2gf：250次迭代结果</summary>

```
--------------------epoch: 249--------------------
Training start ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 138/138 [01:15<00:00,  1.83it/s]
Wait for validation ...
* Acc@1 24.653% Acc@5 35.215%.
* Best Prec@1: 25.050%
Finish !
```
</details>

这些结果的Best Prec都没有Resnet50的那次结果好，所以也没有提交到kaggle上
