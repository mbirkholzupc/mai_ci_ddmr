from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten

from models.core.validation.BestAugmentedModelRunner import AugmentedModelRunner, BaseModelRunner
from tensorflow.keras.applications import MobileNetV2


def model_builder(intermediate_layers):
    model = Sequential(
        [
            Input(shape=(224, 224, 3))
        ],
    )

    mobilenet = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    mobilenet.trainable = True

    print("Number of layers in the base model: ", len(mobilenet.layers))

    # Fine tune from this layer
    fine_tune_at = 100

    for layer in mobilenet.layers[:fine_tune_at]:
        layer.trainable = False

    print('Number of trainable variables = {}'.format(len(mobilenet.trainable_variables)))
    model.add(mobilenet)
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
