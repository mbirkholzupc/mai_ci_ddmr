import shutil
from os import mkdir
from os.path import realpath, dirname, join, exists
from skimage.io import imsave
from loader_splitter.Loader import Loader
from loader_splitter.Splitter import Splitter

if __name__ == "__main__":
    loader = Loader(join(dirname(realpath(__file__)), "./data/train/all"), ["benign", "malignant"])
    data = loader.load()
    folders = ["benign", "malignant"]
    base_folder = "./data/{set}/folds/fold"

    if exists(base_folder.format(set="train")):
        shutil.rmtree(base_folder.format(set="train"))
        mkdir(base_folder.format(set="train"))
    if exists(base_folder.format(set="validation")):
        shutil.rmtree(base_folder.format(set="validation"))
        mkdir(base_folder.format(set="validation"))

    for fold, split in enumerate(Splitter().split(data)):
        file = base_folder + str(fold) + "/{cls}/{id}.jpg"
        for id_image, (img, id_class) in enumerate(zip(split.train_set.X, split.train_set.y)):
            imsave(file.format(set="train", cls=folders[id_class], id=id_image), img)

        for id_image, (img, id_class) in enumerate(zip(split.test_set.X, split.test_set.y)):
            imsave(file.format(set="validation", cls=folders[id_class], id=id_image), img)
