from sklearn.model_selection import StratifiedKFold

from loader_splitter.Split import Split
from loader_splitter.SupervisedData import SupervisedData


class Splitter:
    """
    Allows to split the images in training and test sets
    """
    def __init__(self):
        self._skf = StratifiedKFold(3)

    def split(self, supervised_data: SupervisedData):
        """
        Split the images in training and test sets
        """
        for train_index, test_index in self._skf.split(supervised_data.X, supervised_data.y):
            train_set = SupervisedData(supervised_data.X[train_index], supervised_data.y[train_index])
            test_set = SupervisedData(supervised_data.X[test_index], supervised_data.y[test_index])
            yield Split(train_set, test_set)
