import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime

from models.pretrained.BinaryInceptionV4 import BinaryInceptionV4
import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    acc_per_fold = []
    loss_per_fold = []
    batch_size = 128
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        '../data/original',  # this is the target directory
        target_size=(224, 224),  # all images will be resized
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        '../data/original',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')
    for i in range(3):
        gc.collect()
        # es = tf.keras.callbacks.EarlyStopping(baseline=0.6, patience=10, restore_best_weights=True)
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model = BinaryInceptionV4().get_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_generator,
                steps_per_epoch=2000 // batch_size,
                epochs=10,
                validation_data=validation_generator,
                validation_steps=800 // batch_size,
                callbacks=[tensorboard_callback])

        scores = model.evaluate(validation_generator, verbose=0, steps_per_epoch=2000 // batch_size, steps=800 // batch_size)
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
