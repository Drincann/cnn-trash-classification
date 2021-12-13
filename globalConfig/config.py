import os
from typing import ClassVar

# 模型持久化目录
modelsDir = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'models'))


# 模型 1 配置
class model2Classi:
    # 模型 1 图片缩放尺寸
    imgWidth = 180
    imgHeight = 180
    # 模型 1 目录
    savedDir = os.path.normpath(os.path.join(
        modelsDir, 'model2Classi', 'storage'))
    # 模型 1 持久化文件名
    savedName = 'model2Classi'
    # 模型 1 持久化文件路径
    savedPath = os.path.normpath(os.path.join(savedDir, savedName))
    # 模型 1 分类数
    classNum = 2
    # 模型 1 分类名称
    classNames = ['organic', 'recycle']
    # 数据集根目录
    dataDir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                            '..', 'data', 'model2ClassiDataset'))
    # 单个样本集数量
    batchSize = 32

    # 迭代次数
    epochs = 15
