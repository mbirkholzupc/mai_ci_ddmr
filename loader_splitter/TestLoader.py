from loader_splitter.Loader import Loader
from os.path import realpath, dirname, join


class TestLoader(Loader):
    """
    Allows to loads the images with their labels
    """

    def __init__(self, image_size=(224, 224)):
        categories = ["benign", "malignant"]
        super().__init__(join(dirname(realpath(__file__)), "../data/test/all"), categories, image_size)
