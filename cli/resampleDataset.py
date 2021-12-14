from tensorflow import keras
import tensorflow as tf
import sys
import os
import argparse
import random
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def extname(filename):
    return filename[filename.rindex('.'):]


def checkInput(floders):
    # 检查文件夹是否存在
    for floder in floders:
        if not os.path.exists(floder):
            print("Error: Input floder not exists")
            exit(1)
    # 检查输入的文件夹是否是文件夹
    for floder in floders:
        if not os.path.isdir(floder):
            print("Error: Input floder is not a floder")
            exit(1)

    # 检查输入的文件夹是否为空
    for floder in floders:
        if len(os.listdir(floder)) == 0:
            print("Error: Input floder is empty")
            exit(1)

    # 检查输入的文件夹中是否至少存在一张图片

    for floder in floders:
        for filename in os.listdir(floder):
            if filename.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP')):
                break
        else:
            print(f"Error: Input floder {floder} has no image")
            exit(1)

    # for floder in floders:
    #     for filename in os.listdir(floder):
    #         if extname(filename) in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP']:
    #             break
    # print(f"Error: Input floder {floder} has no image")
    # exit(1)


def main():
    import globalConfig.config as config
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input floder",
                        required=True, type=str, nargs='+')
    args = parser.parse_args()
    checkInput(args.input)

    # 查看所有输入文件中图片最多的数量
    maxCount = 0
    for floder in args.input:
        count = len(os.listdir(floder))
        if count > maxCount:
            maxCount = count

    imgHeight = config.modelMulClassi.imgHeight
    imgWidth = config.modelMulClassi.imgWidth

    process = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal",
                                    input_shape=(imgHeight,
                                                 imgWidth,
                                                 3)),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
        ]
    )

    # 遍历每个小于最大数量的文件夹
    for floder in args.input:
        # 图片列表
        imgList = os.listdir(floder)
        # 过滤图片
        imgList = [filename for filename in imgList if filename.endswith(
            ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP'))]
        # 随机重采样图片直到达到最大数量

        while len(os.listdir(floder)) < maxCount:
            # 随机选择一张图片
            filename = random.choice(imgList)

            img = keras.preprocessing.image.load_img(
                os.path.join(floder, filename),
                target_size=(imgHeight, imgWidth))
            img = keras.preprocessing.image.img_to_array(img)

            # to dataset
            img = tf.expand_dims(img, 0)

            img = process(img)

            # to image
            img = tf.squeeze(img, 0)

            img = keras.preprocessing.image.array_to_img(img)
            img.save(os.path.join(floder, f'{filename}_resample.jpg'))
            print(
                f"{floder} resample {filename} to {filename}_resample.jpg {len(os.listdir(floder))}/{maxCount}")


main()
