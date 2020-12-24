from datetime import datetime

from loader_splitter.Loader import Loader
from loader_splitter.Splitter import Splitter
from models.pretrained.BinaryInceptionV4 import BinaryInceptionV4
from os.path import dirname, realpath, join
import numpy as np
import gc
import tensorflow as tf

def main():
    loader = Loader(join(dirname(realpath(__file__)), "../../data/resized"), ["benign", "malignant"])
    data = loader.load()
    acc_per_fold = []
    loss_per_fold = []
    batch_size = 128
    for fold, split in enumerate(Splitter().split(data)):
        gc.collect()
        es = tf.keras.callbacks.EarlyStopping(baseline=0.6, patience=10, restore_best_weights=True)
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        X_train, y_train = split.train_set.X, split.train_set.y
        X_test, y_test = split.test_set.X, split.test_set.y
        model = BinaryInceptionV4().get_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], validation_data=(X_test, y_test), callbacks=[es, tensorboard_callback])
        model.fit(X_train, y_train, epochs=50, batch_size=batch_size)
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
