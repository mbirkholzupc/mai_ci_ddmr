from keras_preprocessing.image import ImageDataGenerator
from os import listdir
from os.path import dirname, realpath, join


class FoldLoader:
    _default_generator = {"rescale": 1. / 255}

    def __init__(self, image_data_generator_args={}):
        self._train_generator = ImageDataGenerator({**image_data_generator_args, **self._default_generator})
        self._validation_generator = ImageDataGenerator(**self._default_generator)

    def split(self, target_size, batch_size):
        data_folder = join(dirname(realpath(__file__)), "../data")
        folder = join(data_folder, "{set}/folds/{fold}")
        fold_folders = listdir(join(data_folder, "train/folds"))
        for fold_folder in fold_folders:
            train_flow = self._train_generator.flow_from_directory(
                folder.format(set="train", fold=fold_folder),
                target_size=target_size,
                batch_size=batch_size,
                class_mode='binary'
            )
            test_flow = self._validation_generator.flow_from_directory(
                folder.format(set="validation", fold=fold_folder),
                target_size=target_size,
                batch_size=batch_size,
                class_mode='binary'
            )
            yield train_flow, test_flow
