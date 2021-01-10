from tensorflow.keras.optimizers import RMSprop

from models.core.validation.ModelRunner import ModelRunner
from models.scratch.runner import model_builder as scratch_model_builder
import numpy as np


def main():
    model_builder = scratch_model_builder
    best_model_compile_params = {
        "dense": 128,
        "optimizer": RMSprop()
    }
    image_size = (224, 224)
    augmentations = [
        {
            "brightness_range": (.9, 1),
            "horizontal_flip": True
        },
        {
            "horizontal_flip": True, 
            "vertical_flip": True,
            "rotation_range": 0.1,
            "shear_range":0.1
        },
        {
            "horizontal_flip": True,
            "vertical_flip": True,
            "shear_range": 225,
            "zoom_range": 30,
            "rotation_range": .1,
            "brightness_range": (.9, 1.1)
        },
        {
            "zoom_range": .2
        },
        {
            "rotation_range": 40,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "brightness_range": (.9, 1),
            "horizontal_flip": True,
            "vertical_flip": True,
            "zoom_range": 0.2,
            "shear_range": 0.2,
            "fill_mode": "nearest"
        }
    ]
    f1_scores = []
    for augmentation in augmentations:
        model_runner = ModelRunner([model_builder], [best_model_compile_params], [image_size],
                                   augmentation=augmentation)
        _, scores = model_runner.run()
        f1_scores.append(scores["f1_score"])

    best_model_i = np.argmax(f1_scores)
    print("Best augmentation parameters are:")
    print(augmentations[best_model_i])


if __name__ == "__main__":
    main()
