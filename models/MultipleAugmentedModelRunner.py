from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from models.ModelRunner import ModelRunner
from models.pretrained.inceptionv4.runner import model_builder as inception_model_builder
from models.pretrained.resnet50.runner import model_builder as resnet50_model_builder
from models.scratch.runner import model_builder as scratch_model_builder
import numpy as np


def main():
    # TODO: Change to the best performing model (F1 score on test data)
    model_builder = inception_model_builder
    # TODO: Change to the best performing model params
    best_model_compile_params = {
        "dense": 128,
        "dropout": .1,
        "optimizer": Adam(.00001)
    }
    # TODO: Change to (224, 244) if not using InceptionV4
    image_size = (299, 299)
    augmentations = [
        {
            "shear_range": .2,
            "zoom_range": .2,
            "rotation_range": .1,
        },
        {
            "horizontal_flip": True,
            "shear_range": .2,
            "zoom_range": .5,
            "rotation_range": .2,
            "brightness_range": (.9, 1.1),
        },
        # {
        #     "vertical_flip": True,
        #     "shear_range": .2,
        #     "zoom_range": .5,
        #     "rotation_range": .1,
        #     "brightness_range": (.9, 1.1),
        # },
        # {
        #     "horizontal_flip": True,
        #     "vertical_flip": True,
        #     "shear_range": .2,
        #     "zoom_range": .2,
        #     "rotation_range": .2,
        # },
        # {
        #     "horizontal_flip": True,
        #     "vertical_flip": True,
        #     "shear_range": .2,
        #     "zoom_range": .2,
        #     "rotation_range": .1,
        #     "brightness_range": (.9, 1.1),
        # }
    ]
    f1_scores = []
    for augmentation in augmentations:
        model_runner = ModelRunner([model_builder], [best_model_compile_params], [image_size],
                                   augmentation=augmentation)
        _, _, scores, _ = model_runner.run()
        f1_scores.append(scores["f1_score"])

    best_model_i = np.argmax(f1_scores)
    print("Best augmentation parameters are:")
    print(augmentations[best_model_i])


if __name__ == "__main__":
    main()
