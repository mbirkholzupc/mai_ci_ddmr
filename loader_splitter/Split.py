from loader_splitter.SupervisedData import SupervisedData


class Split:
    """
    Stores a single split with train and test sets
    """
    def __init__(self, train_set: SupervisedData, test_set: SupervisedData):
        self._train_set = train_set
        self._test_set = test_set

    @property
    def train_set(self) -> SupervisedData:
        """
        Returns the train set
        """
        return self._train_set

    @property
    def test_set(self) -> SupervisedData:
        """
        Returns the test set
        """
        return self._test_set
