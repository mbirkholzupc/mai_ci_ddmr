from sklearn.model_selection import StratifiedKFold

from loader_splitter.Split import Split
from loader_splitter.SupervisedData import SupervisedData
from sklearn.utils import shuffle


class Splitter:
    """
    Allows to split the images in training and test sets
    """

    def __init__(self):
        self._skf = StratifiedKFold(5, shuffle=True, random_state=43)

    def split(self, supervised_data: SupervisedData):
        """
        Split the images in training and test sets
        """
        for train_index, test_index in self._skf.split(supervised_data.X, supervised_data.y):
            X_train = supervised_data.X[train_index]
            y_train = supervised_data.y[train_index]
            X_test = supervised_data.X[test_index]
            y_test = supervised_data.y[test_index]
            shuffle(X_train, y_train, replace=True)
            shuffle(X_test, y_test, replace=True)
            train_set = SupervisedData(X_train, y_train)
            test_set = SupervisedData(X_test, y_test)
            yield Split(train_set, test_set)
