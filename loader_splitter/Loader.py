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
        samples = np.empty((n_images, 224, 224, 3), dtype=np.uint8)
        labels = []
        image_index = 0
        for i, category in enumerate(self._categories):
            category_path = path.join(self._base_folder, category)
            filenames = listdir(category_path)
            for j, filename in enumerate(filenames):
                image_path = path.join(category_path, filename)
                samples[image_index] = imread(image_path).astype(np.uint8)
                labels.append(self._categories_encoded[i])
                image_index += 1
        return SupervisedData(samples, np.array(labels))
