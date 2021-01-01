from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

from models.BestAugmentedModelRunner import AugmentedModelRunner, BaseModelRunner
from tensorflow.keras.applications import resnet50 as Resnet50


def model_builder(intermediate_layers):
    model = Sequential(
        [
            Input(shape=(224, 224, 3)),
        ],
        name="ResNet50"
    )
    model.add(Resnet50.ResNet50(include_top=False, pooling='avg'))
    for layer in intermediate_layers:
        model.add(layer)
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
