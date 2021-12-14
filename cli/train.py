import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import sys
import os
import argparse
from ModelsEnum import Models


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def trainModel2Classi():
    import globalConfig.config as config
    dataDir = pathlib.Path(config.model2Classi.dataDir)
    imgHeight = config.model2Classi.imgHeight
    imgWidth = config.model2Classi.imgWidth
    batchSize = config.model2Classi.batchSize
    dataDir = pathlib.Path(config.model2Classi.dataDir)
    epochs = config.model2Classi.epochs

    trainDataset = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(imgHeight, imgWidth),
        batch_size=batchSize
    )

    valDataset = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(imgHeight, imgWidth),
        batch_size=batchSize
    )

    trainDataset = trainDataset.cache().shuffle(
        1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    valDataset = valDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    from models.model2Classi.modelDefinition import model

    print(model.summary())

    history = model.fit(
        trainDataset,
        validation_data=valDataset,
        epochs=epochs
    )

    acc = history.history['accuracy']
    valAcc = history.history['val_accuracy']

    loss = history.history['loss']
    valLoss = history.history['val_loss']

    epochsRange = range(epochs)

    modelConfig = config.modelMulClassi
    if not os.path.exists(modelConfig.savedDir):
        os.makedirs(modelConfig.savedDir)

    model.save_weights(os.path.join(
        modelConfig.savedDir, modelConfig.savedName))

    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochsRange, acc, label='training')
    plt.plot(epochsRange, valAcc, label='validation')
    plt.legend(loc='lower right')
    plt.title('accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochsRange, loss, label='training')
    plt.plot(epochsRange, valLoss, label='validation')
    plt.legend(loc='upper right')
    plt.title('loss')
    plt.show()


def trainModelMulClassi():
    import globalConfig.config as config
    dataDir = pathlib.Path(config.modelMulClassi.dataDir)
    imgHeight = config.modelMulClassi.imgHeight
    imgWidth = config.modelMulClassi.imgWidth
    batchSize = config.modelMulClassi.batchSize
    dataDir = pathlib.Path(config.modelMulClassi.dataDir)
    epochs = config.modelMulClassi.epochs

    trainDataset = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(imgHeight, imgWidth),
        batch_size=batchSize
    )

    valDataset = tf.keras.utils.image_dataset_from_directory(
        dataDir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(imgHeight, imgWidth),
        batch_size=batchSize
    )

    trainDataset = trainDataset.cache().shuffle(
        1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    valDataset = valDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    from models.modelMulClassi.modelDefinition import model

    print(model.summary())

    history = model.fit(
        trainDataset,
        validation_data=valDataset,
        epochs=epochs
    )

    acc = history.history['accuracy']
    valAcc = history.history['val_accuracy']

    loss = history.history['loss']
    valLoss = history.history['val_loss']

    epochsRange = range(epochs)

    modelConfig = config.modelMulClassi
    if not os.path.exists(modelConfig.savedDir):
        os.makedirs(modelConfig.savedDir)

    model.save_weights(os.path.join(
        modelConfig.savedDir, modelConfig.savedName))

    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochsRange, acc, label='training')
    plt.plot(epochsRange, valAcc, label='validation')
    plt.legend(loc='lower right')
    plt.title('accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochsRange, loss, label='training')
    plt.plot(epochsRange, valLoss, label='validation')
    plt.legend(loc='upper right')
    plt.title('loss')
    plt.show()


def main():
    models = [Models.Model2Classi, Models.ModelMulClassi]
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model to training", required=False,
                        default='2classi', type=str, choices=models)

    args = parser.parse_args()

    if args.model == Models.Model2Classi.value:
        trainModel2Classi()
    elif args.model == Models.ModelMulClassi.value:
        trainModelMulClassi()
    else:
        raise ValueError('Model not found')


main()
