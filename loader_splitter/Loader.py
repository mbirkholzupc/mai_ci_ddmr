from os import path, listdir
from skimage.io import imread
import numpy as np

from loader_splitter.SupervisedData import SupervisedData


class Loader:
    """
    Allows to loads the images with their labels
    """

    def __init__(self, base_folder: str, categories: list):
        self._base_folder = base_folder
        self._categories = categories

    def load(self) -> SupervisedData:
        """
        Loads the images with their labels
        """
        samples = []
        labels = []
        for category in self._categories:
            category_path = path.join(self._base_folder, category)
            for filename in listdir(category_path):
                image_path = path.join(category_path, filename)
                samples.append(np.array(imread(image_path)))
                labels.append(category)
        return SupervisedData(np.array(samples), np.array(labels))
