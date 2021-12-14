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


def predict(model, imgPath, globalConfig):
    # 如果 imgPath 是数组
    if isinstance(imgPath, list):
        return [predict(model, img, globalConfig) for img in imgPath]

    img = numpy.array(
        Image.open(imgPath)
        .convert('RGB')
        .resize((globalConfig.model2Classi.imgHeight, globalConfig.model2Classi.imgWidth))
    )
    img = tensorflow.expand_dims(img, 0)
    return tensorflow.nn.softmax(model.predict(img))


def main():
    models = [Models.Model2Classi, Models.ModelMulClassi]
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input image",
                        required=True, type=str, nargs='+')
    parser.add_argument("-m", "--model", help="Model to use", required=False,
                        default='2classi', type=str, choices=models)

    args = parser.parse_args()
    import globalConfig.config as config
    from models.model2Classi.modelDefinition import model as model2Classi
    # from ..models.modelMulClassi.modelDefinition import model as modelMulClassi
    if args.model == Models.Model2Classi.value:
        model = model2Classi
        model.load_weights(config.model2Classi.savedPath)
    # elif args.model == Models.ModelMulClassi.value:
    #     model = modelMulClassi
    #     modelMulClassi.load_weights(config.modelMulClassi.savedPath)
    else:
        raise ValueError('Model not found')

    scores = predict(model, args.input, config)

    print()
    for i, score in enumerate(scores):
        print(f'{args.input[i]} probability: {score[0]}')
        print(pandas.Series({classification: f'{str(float(score[0][i])*100)} %' for i, classification in enumerate(
            config.model2Classi.classNames)}))
        print(f'result: {config.model2Classi.classNames[numpy.argmax(score)]}')
        print()


main()
