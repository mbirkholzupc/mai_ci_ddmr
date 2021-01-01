from keras_preprocessing.image import ImageDataGenerator
from os import listdir
from os.path import dirname, realpath, join

from loader_splitter.Loader import Loader


class FoldLoader:
    def __init__(self, image_data_generator_args=None):
        if image_data_generator_args is None:
            image_data_generator_args = {}
        self._train_generator = ImageDataGenerator(**image_data_generator_args) if bool(image_data_generator_args) \
            else None

    def split(self, target_size, batch_size):
        data_folder = join(dirname(realpath(__file__)), "../data")
        folder = join(data_folder, "{set}/folds/{fold}")
        fold_folders = listdir(join(data_folder, "train/folds"))
        for fold_folder in fold_folders:
            train_folder = folder.format(set="train", fold=fold_folder)
            if not bool(self._train_generator):
                train_loader = Loader(train_folder, ["benign", "malignant"], target_size)
                train_data = train_loader.load()
            else:
                train_data = self._train_generator.flow_from_directory(
                    train_folder,
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode='binary'
                )
            test_loader = Loader(folder.format(set="validation", fold=fold_folder), ["benign", "malignant"],
                                 target_size)
            test_data = test_loader.load()
            yield train_data, test_data
