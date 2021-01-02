import shutil
import pathlib
from os.path import realpath, dirname, join, exists

from skimage import img_as_ubyte
from skimage.io import imsave
from loader_splitter.Loader import Loader
from loader_splitter.Splitter import Splitter

if __name__ == "__main__":
    loader = Loader(join(dirname(realpath(__file__)), "./data/train/all"), ["benign", "malignant"])
    data = loader.load()
    folders = ["benign", "malignant"]
    base_folder = "./data/{set}/folds"

    if exists(base_folder.format(set="train")):
        shutil.rmtree(base_folder.format(set="train"))
        pathlib.Path(base_folder.format(set="train")).mkdir(parents=True, exist_ok=True)
    if exists(base_folder.format(set="validation")):
        shutil.rmtree(base_folder.format(set="validation"))
        pathlib.Path(base_folder.format(set="validation")).mkdir(parents=True, exist_ok=True)

    for fold, split in enumerate(Splitter().split(data)):
        folder = base_folder + "/fold" + str(fold) + "/{cls}"
        file = folder + "/{id}.jpg"
        for id_image, (img, id_class) in enumerate(zip(split.train_set.X, split.train_set.y)):
            if not exists(folder.format(set="train", cls=folders[id_class])):
                pathlib.Path(folder.format(set="train", cls=folders[id_class])).mkdir(parents=True, exist_ok=True)
            imsave(file.format(fold=fold, set="train", cls=folders[id_class], id=id_image), img_as_ubyte(img))

        for id_image, (img, id_class) in enumerate(zip(split.test_set.X, split.test_set.y)):
            if not exists(folder.format(set="validation", cls=folders[id_class])):
                pathlib.Path(folder.format(set="validation", cls=folders[id_class])).mkdir(parents=True, exist_ok=True)
            imsave(file.format(fold=fold, set="validation", cls=folders[id_class], id=id_image), img_as_ubyte(img))
