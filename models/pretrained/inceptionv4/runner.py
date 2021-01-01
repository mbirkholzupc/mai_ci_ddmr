from models.AugmentedModelRunner import AugmentedModelRunner, BaseModelRunner
from models.pretrained.inceptionv4.BinaryInceptionV4 import BinaryInceptionV4


def model_builder():
    return BinaryInceptionV4().get_model()


def main(augmented=False):
    cls = AugmentedModelRunner if augmented else BaseModelRunner
    model_runner = cls(
        model_builder,
        image_size=(299, 299)
    )
    model_runner.run()


if __name__ == "__main__":
    main()
