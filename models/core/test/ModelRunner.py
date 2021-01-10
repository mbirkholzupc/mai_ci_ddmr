from os.path import join, dirname, realpath

from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from models.core.validation.ModelRunner import ModelRunner as ValidationModelRunner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from loader_splitter.TestLoader import TestLoader
from loader_splitter.TrainLoader import TrainLoader


class ModelRunner(ValidationModelRunner):
    def __init__(self, model_builder, model_compile_params=None, image_size=(224, 224),
                 batch_size=128, augmentation=None):
        super().__init__([model_builder], [model_compile_params], [image_size], batch_size, augmentation)

    def run(self):
        if bool(self._augmentation):
            train_data_generator = ImageDataGenerator(
                **{**self._augmentation, **{"rescale": 1. / 255, "dtype": np.float16}}
            )
            train_data = train_data_generator.flow_from_directory(
                join(dirname(realpath(__file__)), "../../../data/train/all"),
                target_size=self._image_sizes[0],
                batch_size=self._batch_size,
                class_mode='binary',
                seed=43
            )
        else:
            train_data = TrainLoader(self._image_sizes[0]).load()
        test_data = TestLoader(self._image_sizes[0]).load()
        best_model, scores, history = self._compile_fit_model(
            self._model_builders[0], self._model_compile_params[0], train_data, test_data)
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(f'Final model scores:')
        self._report_scores(scores)
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

        # F1 score
        f1_score = 2 * np.array(history.history['precision']) * np.array(history.history['recall']) / \
                   (np.array(history.history['precision']) + np.array(history.history['recall']))
        val_f1_score = 2 * np.array(history.history['val_precision']) * np.array(history.history['val_recall']) / \
                       (np.array(history.history['val_precision']) + np.array(history.history['val_recall']))
        plt.plot(f1_score)
        plt.plot(val_f1_score)
        plt.title('model f1 score')
        plt.ylabel('f1 score')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('f1_score.png')
        plt.show()
        # Loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('loss.png')
        plt.show()
        # Confusion matrix
        y_pred = best_model.predict(test_data.X).flatten()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        categories = ["benign", "malignant"]
        df_cm = pd.DataFrame(confusion_matrix(test_data.y, y_pred),
                             index=categories, columns=categories)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        fig = plt.figure(figsize=(4, 4))
        fig.suptitle("Confusion matrix")
        _ = sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="d")
        plt.savefig('cm.png')
        plt.show()
        best_model.save('model.h5')

        return best_model, 0, scores, history
