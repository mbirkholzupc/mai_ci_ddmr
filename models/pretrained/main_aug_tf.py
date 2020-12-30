import os

from loader_splitter.FoldLoader import FoldLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime

from models.pretrained.BinaryInceptionV4 import BinaryInceptionV4
import gc
import numpy as np
import tensorflow as tf


def main():
    acc_per_fold = []
    loss_per_fold = []
    batch_size = 128
    target_size = (299, 299)
    fold_loader = FoldLoader({"shear_range":0.2,
        "zoom_range":0.2,
        "rotation_range":35,
        "horizontal_flip":0.7})

    for i, (train_generator, validation_generator) in enumerate(fold_loader.split(target_size, batch_size)):
        gc.collect()
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model = BinaryInceptionV4().get_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_generator,
                  epochs=50,
                  validation_data=validation_generator,
                  callbacks=[tensorboard_callback]
                  )

        scores = model.evaluate(validation_generator, verbose=0)
        print(
            f'Score for fold {i}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
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
