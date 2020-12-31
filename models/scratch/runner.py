from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten

from models.AugmentedModelRunner import AugmentedModelRunner, BaseModelRunner


def model_builder():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

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
