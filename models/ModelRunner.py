import os
import numpy as np
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from tensorflow.python.keras.layers import Dense, Dropout

from loader_splitter.FoldLoader import FoldLoader
from loader_splitter.SupervisedData import SupervisedData
from loader_splitter.TestLoader import TestLoader
from loader_splitter.TrainLoader import TrainLoader
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from abc import ABC


class ModelRunner(ABC):
    def __init__(self, model_builders, model_compile_params=None, image_size=(224, 224),
                 batch_size=128, augmentation=None, show_extra_details=False):
        if augmentation is None:
            augmentation = {}
        if model_compile_params is None:
            model_compile_params = []
        self._model_builders = model_builders
        self._model_compile_params = model_compile_params
        self._image_size = image_size
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._show_extra_details = show_extra_details

    def run(self):
        fold_loader = FoldLoader(self._augmentation)
        avg_scores_per_model = []
        for model, model_builder in enumerate(self._model_builders()):
            print(f">>> Model {model + 1}")
            scores_per_fold = []
            model_compile_params = {} if len(self._model_compile_params) <= model else \
                self._model_compile_params[model]
            for fold, (train_data, validation_data) in \
                    enumerate(fold_loader.split(self._image_size, self._batch_size)):
                model = model_builder()
                scores, _ = self._compile_fit_model(model, model_compile_params, train_data, validation_data)
                scores_per_fold.append(scores)

            accum_f1_score = 0
            accum_accuracy = 0
            accum_precision = 0
            accum_recall = 0
            accum_epochs = 0
            for scores in scores_per_fold:
                accum_f1_score += scores["f1_score"]
                accum_accuracy += scores["accuracy"]
                accum_precision += scores["precision"]
                accum_recall += scores["recall"]
                accum_epochs += scores["epochs"]
            avg_score_per_fold = {
                "f1_score": round(accum_f1_score / len(scores_per_fold), 4),
                "accuracy": round(accum_accuracy / len(scores_per_fold), 4),
                "precision": round(accum_precision / len(scores_per_fold), 4),
                "recall": round(accum_recall / len(scores_per_fold), 4),
                "epochs": accum_epochs // len(scores_per_fold)
            }
            print('------------------------------------------------------------------------')
            print('Average model score')
            self._report_scores(avg_score_per_fold)
            print('------------------------------------------------------------------------')
            avg_scores_per_model.append(avg_score_per_fold)

        best_model_i = np.argmin([score["f1_score"] for score in avg_scores_per_model])
        best_model_scores = avg_scores_per_model[best_model_i]
        print('===========================================================================')
        print(f'Best model is nº {best_model_i} with score:')
        self._report_scores(best_model_scores)
        print('===========================================================================')
        print("Doing final training and test")
        best_model_compile_params = {} if best_model_i >= len(self._model_compile_params) else \
            self._model_compile_params[best_model_i]
        best_model = self._model_builders(best_model_i)()
        train_data = TrainLoader().load()
        test_data = TestLoader().load()
        scores, history = self._compile_fit_model(
            best_model, best_model_compile_params, train_data, test_data)
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(f'Final model scores:')
        self._report_scores(scores)
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        if self._show_extra_details:
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
            plt.savefig(f'f1_score_{best_model_i}.png')
            plt.show()
            # Loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig(f'loss_{best_model_i}.png')
            plt.show()
            # Confusion matrix
            cm = confusion_matrix(test_data.y, best_model.predict(test_data.X))
            plot_confusion_matrix(cm, class_names=["benign", "malignant"])
            plt.savefig(f'cm_{best_model_i}.png')
            plt.show()
        best_model.save(f'model_{best_model_i}.h5')

    @staticmethod
    def _compile_fit_model(model, model_compile_params, train_data, validation_data):
        if "dense" in model_compile_params and model_compile_params["dense"] > 0:
            classifier = Dense(model_compile_params["dense"], activation='relu')(model.layers[-2].output)
            del model_compile_params["dense"]
            if "dropout" in model_compile_params and model_compile_params["dropout"] > 0:
                dropout = Dropout(model_compile_params["dropout"])(classifier)
                del model_compile_params["dropout"]
                model.layers[-1].output(dropout)
            else:
                model.layers[-1].output(classifier)
        elif "dropout" in model_compile_params and model_compile_params["dropout"] > 0:
            dropout = Dropout(model_compile_params["dropout"])(model.layers[-2].output)
            del model_compile_params["dropout"]
            model.layers[-1].output(dropout)

        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, baseline=0.6,
                                              restore_best_weights=True)
        learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                       patience=5,
                                                                       verbose=1,
                                                                       factor=0.5,
                                                                       min_lr=1e-7)
        model.compile(**{**model_compile_params,
                         **{"loss": "binary_crossentropy", "metrics": ["accuracy",
                                                                       tf.keras.metrics.Precision(name="precision"),
                                                                       tf.keras.metrics.Recall(name="recall")]}})
        base_fit_params = {"epochs": 1, "validation_data": (validation_data.X, validation_data.y),
                           "callbacks": [es, learning_rate_reduction]}

        if type(train_data) is SupervisedData:
            extra_fit_params = {"x": train_data.X, "y": train_data.y}
        else:
            extra_fit_params = {"x": train_data}

        history = model.fit(**{**base_fit_params, **extra_fit_params})
        scores = model.evaluate(x=validation_data.X, y=validation_data.y, verbose=0)
        f1_score = 2 * scores[1] * scores[2] / (scores[1] + scores[2])
        return {"f1_score": f1_score, "accuracy": scores[0],
                "precision": scores[1], "recall": scores[2],
                "epochs": len(history.history['loss'])}, history

    @staticmethod
    def _report_scores(scores):
        print('F1 score: ', scores["f1_score"])
        print('Accuracy: ', scores["accuracy"])
        print('Precision: ', scores["precision"])
        print('Recall: ', scores["recall"])
        print('Epochs: ', scores["epochs"])
