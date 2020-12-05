from os import path, listdir
from skimage.io import imread
import numpy as np
from sklearn.preprocessing import LabelEncoder

from loader_splitter.SupervisedData import SupervisedData


class Loader:
    """
    Allows to loads the images with their labels
    """

    def __init__(self, base_folder: str, categories: list):
        self._base_folder = base_folder
        self._categories = categories
        self._categories_encoded = LabelEncoder().fit_transform(self._categories)

    def load(self) -> SupervisedData:
        """
        Loads the images with their labels
        """
        n_images = sum([len(listdir(path.join(self._base_folder, category))) for category in self._categories])
        samples = np.empty((n_images, 299, 299, 3), np.float16)
        labels = []
        for i, category in enumerate(self._categories):
            category_path = path.join(self._base_folder, category)
            for j, filename in enumerate(listdir(category_path)):
                image_index = i * len(self._categories) + j
                image_path = path.join(category_path, filename)
                samples[image_index] = np.divide(np.array(imread(image_path), dtype=np.float16), 255.0)
                labels.append(self._categories_encoded[i])
        return SupervisedData(samples, np.array(labels))
