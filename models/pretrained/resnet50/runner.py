from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten

from models.BestAugmentedModelRunner import AugmentedModelRunner, BaseModelRunner
from tensorflow.keras.applications import resnet50 as Resnet50


def model_builder(intermediate_layers):
    model = Sequential(
        [
            Input(shape=(224, 224, 3)),
        ],
        name="ResNet50"
    )

    resnet = Resnet50.ResNet50(include_top=False)

    # Freeze resnet layers
    # See https://radiant-brushlands-42789.herokuapp.com/towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38
    for layer in resnet.layers:
        layer.trainable = layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']

    model.add(resnet)
    model.add(Flatten())

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
