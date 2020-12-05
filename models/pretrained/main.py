from loader_splitter.Loader import Loader
from loader_splitter.Splitter import Splitter
from models.pretrained.BinaryInceptionV4 import BinaryInceptionV4
from os.path import dirname, realpath, join
import numpy as np


def main():
    loader = Loader(join(dirname(realpath(__file__)), "../../data/resized"), ["benign", "malignant"])
    data = loader.load()
    acc_per_fold = []
    loss_per_fold = []
    batch_size = 128
    for fold, split in enumerate(Splitter().split(data)):
        X_train, y_train = split.train_set.X, split.train_set.y
        X_test, y_test = split.test_set.X, split.test_set.y
        model = BinaryInceptionV4().get_model()
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_train, y_train, epochs=3, batch_size=batch_size)
        scores = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)
        print(
            f'Score for fold {fold}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


if __name__ == "__main__":
    main()
