# 卷积神经网络垃圾分类应用

这是一个使用 `TensorFlow` 实现的，基于卷积神经网络的垃圾分类模型。

> 这是 -> 软件工程，大三上学期，河北农业大学，理工学院，齐立萍老师的科学计算与数据分析课程的课程设计。

该应用提供了：

- 带有大量文本的 jupyter notebook
- 数据集预处理脚本
- 用于训练的命令行工具
- 用于预测新数据的命令行工具

> ! 该仓库不包含数据集，也未包含训练好的神经网络权重持久化文件。

继续阅读，来准备数据集，并训练模型，生成可用于 `cli/predict.py` 这个用于预测的命令行工具的训练好的神经网络权重持久化文件。

再开始之前，确保你现在正处于该仓库的根目录：

```sh
git clone https://github.com/Drincann/cnn-trash-classification.git
cd cnn-trash-classification
```

## 准备你的环境

```sh
python3 -m pip install -r requirements.txt
```

注意，请使用 TensorFlow >= 2.7.0，来支持一些该 repo 使用到的新特性。

请不要使用 macOS，Apple 目前对 TensorFlow 的支持不完整，见 [https://github.com/apple/tensorflow_macos](https://github.com/apple/tensorflow_macos)。

使用 Windows 或 Ubuntu Linux。

## Dataset

### 选择数据集

#### 两个标签的数据集

感谢 [梨为乐](https://www.heywhale.com/home/user/profile/6137612c1cac2c04682f9c47) 的 [有机、可回收物品数据集](https://www.heywhale.com/mw/project/619b91bc7d74800017258af5/dataset)。

你可以在这里下载它：

[https://www.heywhale.com/mw/project/619b91bc7d74800017258af5/dataset](https://www.heywhale.com/mw/project/619b91bc7d74800017258af5/dataset)

#### 更多标签的数据集

感谢这个 repo -> [garythung/trashnet](https://github.com/garythung/trashnet) 提供的更多标签的数据集。

你可以在这里找到 google 云盘上的两个数据集（需要梯子）：

[https://github.com/garythung/trashnet](https://github.com/garythung/trashnet)

### 准备数据集

请在根目录创建 `data` 目录，脚本将在这里寻找数据集。

对于上节中的第一个数据集，它应该位于 `./data/model2ClassiDataset`：

```sh
mkdir -p ./data/model2ClassiDataset
```

将第一个数据集解压：

```sh
$ tree -L 2
.
├── TEST
│   ├── O
│   └── R
└── TRAIN
    ├── O
    └── R
```

`"O"` 指的是有机物(organic)，`"R"` 值的是可回收物(recycle)。

数据集的提供者已经为我们分好了测试集和训练集，但在后面的过程中，我们要自己拆分数据集。

所以，现在你可以选择仅使用测试集、训练集中的一个，或者将它们合并，使用整个数据集。

然后，重命名 `O` 为 `organic`，对应的，`R` 重命名为 `recycle`，将它们放到 `./data/model2ClassiDataset/` 目录下：

```sh
cp -R ${datasetroot}/organic ./data/model2ClassiDataset/
cp -R ${datasetroot}/recycle ./data/model2ClassiDataset/
```

这是因为，在后面进行数据集拆分时，脚本会将文件名作为标签名，而且，在全局配置文件 `./globalConfig/config.py` 中我们也是这样配置的。

## Train

### 两标签数据集

你可以参考 jupyter notebook `./notebookModel2Classi.ipynb`，它是可运行的，并带有大量描述文本，流程包括从读取数据集到最终输出持久化的卷积神经网络权重文件。

或者使用脚本训练：

```sh
python3 ./cli/train.py -m 2classi
```

`-m` 参数指定要使用什么数据集训练模型，可选项 `['mulclassi', '2classi']`，`2classi` 是默认值。

训练是一个漫长的过程，

在仅使用训练集提供的 2k+ 个图片的情况下，在连接电源的移动平台上使用 CPU i5-9300H 进行训练，花费时间大概 10min+。

如果要使用 GPU 加速训练，请访问 [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) 查看你的 GPU 支持情况，并按照文档安装环境依赖。

你可以使用这种方式查看可用的计算资源：

```python
gpus = tf.config.list_physical_devices(device_type='GPU')
cpus = tf.config.list_physical_devices(device_type='CPU')
print(gpus, cpus)
```

在连接电源的移动平台上使用 RTX 2060 进行训练，花费时间大概 1min 左右。

在使用命令行工具进行训练结束后，会写出已训练好的卷积神经网络权重持久化文件，三个文件会被写到 `./models/model2Classi/storage` 下。

这些持久化文件被用于命令行工具 `./cli/predict.py`，这个命令行工具通过重新加载卷积神经网络权重来预测输入的数据。

### 多标签数据集

not implement

## Predict

接下来使用 `./cli/predict.py` 进行预测。

你可以使用 `-h` 参数来查看帮助信息：

```sh
python3 ./cli/predict.py -h
```

使用示例：

```sh
python3 ./cli/predict.py -i trash.jpg -m 2classi
```

`-i` 用来指定多个要预测的图片路径输入（也可以是一个文件）。

`-m` 指定使用哪个已训练好的模型，可选项 `['mulclassi', '2classi']`，`2classi` 是默认值。

然后，对于使用两标签数据集训练的模型，该脚本会分别对每个输入图片输出两个标签的概率，以及一个结果。
