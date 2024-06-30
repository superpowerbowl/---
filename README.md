# 任务一介绍
自选数据集STL-10，使用SimCLR自监督学习算法，之后在CIFAR-100上使用Linear Classification Protocol进行线性分类器层的训练，并进行模型测试。基座模型为resnet-18模型，并对比在CIFAR-100数据集上进行自监督学习以及从零开始进行监督学习之间效果的改变。

## 数据准备
下载CIFAR-100, STL-10数据集，数据集下载链接如下：

CIFAR-100:

https://www.cs.toronto.edu/~kriz/cifar.html

STL-10:

https://cs.stanford.edu/~acoates/stl10/

下载数据后，在根目录下创建datasets文件夹，将下载好的数据集解压到datasets文件夹中，文件目录格式如下:
```
root
├── datasets
│   ├── cifar-10-batches-py
│   ├── cifar-100-python
│   │   ├── file.txt~
│   │   ├── meta
│   │   ├── test
│   │   └── train
│   └── stl10_binary
│       ├── class_names.txt
│       ├── fold_indices.txt
│       ├── test_X.bin
│       ├── test_y.bin
│       ├── train_X.bin
│       ├── train_y.bin
│       └── unlabeled_X.bin
└── cifar-100-python.tar.gz
└── stl10_binary.tar.gz
```

## 自监督学习+微调训练
### 自监督学习
运行
```
python train.py 
```

### 在CIFAR-100上利用线性分类协议微调线性分类器层
运行
```
python finetune.py
```

## 从零开始监督学习
运行
```
python supervised_learning_CIFAR100.py
```

