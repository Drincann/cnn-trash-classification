import tensorflow
from PIL import Image
import numpy
import argparse
import os
import sys
import pandas
from ModelsEnum import Models

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def predict(model, imgPath, imgHeight, imgWidth):
    # 如果 imgPath 是数组
    if isinstance(imgPath, list):
        return [predict(model, img,  imgHeight, imgWidth) for img in imgPath]

    img = numpy.array(
        Image.open(imgPath)
        .convert('RGB')
        .resize((imgHeight, imgWidth))
    )
    img = tensorflow.expand_dims(img, 0)
    return tensorflow.nn.softmax(model.predict(img))


def main():
    models = [Models.Model2Classi.value, Models.ModelMulClassi.value]
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input image",
                        required=True, type=str, nargs='+')
    parser.add_argument("-m", "--model", help="Model to use", required=False,
                        default='2classi', type=str, choices=models)

    args = parser.parse_args()
    import globalConfig.config as config
    from models.model2Classi.modelDefinition import model as model2Classi
    from models.modelMulClassi.modelDefinition import model as modelMulClassi
    if args.model == Models.Model2Classi.value:
        model = model2Classi
        classNames = config.model2Classi.classNames
        model.load_weights(config.model2Classi.savedPath)
        imgHeight = config.model2Classi.imgHeight
        imgWidth = config.model2Classi.imgWidth
    elif args.model == Models.ModelMulClassi.value:
        model = modelMulClassi
        classNames = config.modelMulClassi.classNames
        modelMulClassi.load_weights(config.modelMulClassi.savedPath)
        imgHeight = config.modelMulClassi.imgHeight
        imgWidth = config.modelMulClassi.imgWidth
    else:
        raise ValueError('Model not found')

    # 检查输入的图片是否是存在
    for img in args.input:
        if not os.path.exists(img):
            print(f"Error: Input img {img} not exists")
            exit(1)

    # 检查输入的图片是否是文件
    for img in args.input:
        if not os.path.isfile(img):
            print(f"Error: Input img {img} is not a file")
            exit(1)

    scores = predict(model, args.input, imgHeight, imgWidth)

    print()
    for i, score in enumerate(scores):
        print(f'{args.input[i]} probability: {score[0]}')
        print(pandas.Series({classification: f'{str(float(score[0][i])*100)} %' for i, classification in enumerate(
            classNames)}))
        print(f'result: {classNames[numpy.argmax(score)]}')
        print()


main()
