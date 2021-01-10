from tensorflow.keras.optimizers import Adam

from models.core.test.ModelRunner import ModelRunner
from models.scratch.runner import model_builder as scratch_model_builder
from models.pretrained.inceptionv4.runner import model_builder as inceptionv4_model_builder
from models.pretrained.resnet50.runner import model_builder as resnet50_model_builder

def main():
    model_builder = scratch_model_builder
    best_model_compile_params = {
        "dense": 128,
        "optimizer": Adam(0.0001)
    }
    image_size = (224, 224)
    augmentation = {
        "brightness_range": (.9, 1),
        "horizontal_flip": True
    }

    model_runner = ModelRunner(model_builder, best_model_compile_params, image_size,
                               augmentation=augmentation)
    model_runner.run()


if __name__ == "__main__":
    main()
