from models.BaseModelRunner import BaseModelRunner


class AugmentedModelRunner(BaseModelRunner):
    def __init__(self, model_builder, image_size=(224, 224), batch_size=128):
        super().__init__(model_builder, image_size, batch_size, self._get_augmentation(), True)

    @staticmethod
    def _get_augmentation():
        return {
            "horizontal_flip": True,
            "vertical_flip": True,
            "shear_range": .2,
            "zoom_range": .2,
            "rotation_range": .1,
            "brightness_range": (.9, 1.1),
        }

