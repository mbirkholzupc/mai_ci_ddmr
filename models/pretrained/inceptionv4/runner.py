from models.AugmentedModelRunner import AugmentedModelRunner, BaseModelRunner
from models.pretrained.inceptionv4.BinaryInceptionV4 import BinaryInceptionV4


def model_builders(index=None):
    def binary_inceptionv4():
        return BinaryInceptionV4().get_model()

    models = [binary_inceptionv4]
    return models[index] if index is not None else models


def main(augmented=False):
    cls = AugmentedModelRunner if augmented else BaseModelRunner
    model_runner = cls(
        model_builders,
        image_size=(299, 299)
    )
    model_runner.run()


if __name__ == "__main__":
    main()
