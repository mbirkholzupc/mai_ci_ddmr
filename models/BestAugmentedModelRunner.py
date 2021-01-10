from models.BaseModelRunner import BaseModelRunner


class AugmentedModelRunner(BaseModelRunner):
    def __init__(self, model_builder, image_size=(224, 224), batch_size=128):
        super().__init__(model_builder, image_size, batch_size, self._get_augmentation(), True, True)

    # TODO: Change the following params as the best obtained in MultipleAugmentedModelRunner
    @staticmethod
    def _get_augmentation():
        return {
            "brightness_range": (.9, 1),
            "horizontal_flip": True
        }

