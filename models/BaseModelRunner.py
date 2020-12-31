from keras.optimizers import Adam, SGD, rmsprop

from models.ModelRunner import ModelRunner


class BaseModelRunner(ModelRunner):
    def __init__(self, model_builder, image_size=(224, 224), batch_size=128, augmentation=None,
                 show_extra_details=False):
        params, model_builders = self._get_model_compile_params_model_builders(model_builder)
        super().__init__(model_builders, params, image_size, batch_size, augmentation, show_extra_details)

    @staticmethod
    def _get_model_compile_params_model_builders(model_builder):
        params = [
            {"dropout": .3, "optimizer": Adam(.001)},
            {"optimizer": SGD(.005)},
            {"dense": 128, "optimizer": rmsprop(.0001)},
            {"dense": 128, "dropout": .3, "optimizer": Adam(.001)},
            {"dense": 128, "dropout": .1, "optimizer": Adam(.00001)}
        ]
        model_builders = [model_builder] * len(params)
        return params, model_builders
