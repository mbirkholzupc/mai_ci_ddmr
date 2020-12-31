from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Dense

from models.AugmentedModelRunner import AugmentedModelRunner, BaseModelRunner
from tensorflow.keras.applications import resnet50 as Resnet50


def model_builder():
    model = Sequential(
        [
            Input(shape=(224, 224, 3)),
        ],
        name="ResNet50"
    )
    model.add(Resnet50.ResNet50(include_top=False, pooling='avg'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def main(augmented=False):
    cls = AugmentedModelRunner if augmented else BaseModelRunner
    model_runner = cls(
        model_builder,
        image_size=(224, 224)
    )
    model_runner.run()


if __name__ == "__main__":
    main()
