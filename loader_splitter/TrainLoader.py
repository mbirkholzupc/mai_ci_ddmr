from loader_splitter.Loader import Loader
from os.path import realpath, dirname, join


class TrainLoader(Loader):
    """
    Allows to loads the images with their labels
    """

    def __init__(self):
        categories = ["benign", "malignant"]
        super().__init__(join(dirname(realpath(__file__)), "../data/train/all"), categories)
