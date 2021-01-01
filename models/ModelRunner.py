import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
from tensorflow.keras.layers import Dense, Dropout

from loader_splitter.FoldLoader import FoldLoader
from loader_splitter.SupervisedData import SupervisedData
from loader_splitter.TestLoader import TestLoader
from loader_splitter.TrainLoader import TrainLoader
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class ModelRunner():
    def __init__(self, model_builders, model_compile_params=None, image_sizes=(224, 224),
                 batch_size=128, augmentation=None, show_extra_details=False):
        if augmentation is None:
            augmentation = {}
        if model_compile_params is None:
            model_compile_params = []
        self._model_builders = model_builders
        self._model_compile_params = model_compile_params
        self._image_sizes = image_sizes
        self._batch_size = batch_size
        self._augmentation = augmentation
        self._show_extra_details = show_extra_details

    def run(self):
        fold_loader = FoldLoader(self._augmentation)
        avg_scores_per_model = []
        for model, (model_builder, image_size) in enumerate(zip(self._model_builders, self._image_sizes)):
            print(f">>> Model {model + 1}")
            scores_per_fold = []
            model_compile_params = {} if len(self._model_compile_params) <= model else \
                self._model_compile_params[model]
            for fold, (train_data, validation_data) in \
                    enumerate(fold_loader.split(image_size, self._batch_size)):
                _, scores, _ = self._compile_fit_model(model_builder, dict(model_compile_params), train_data, validation_data)
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

        best_model_i = np.argmax([score["f1_score"] for score in avg_scores_per_model])
        best_model_scores = avg_scores_per_model[best_model_i]
        print('===========================================================================')
        print(f'Best model is nº {best_model_i} with score:')
        self._report_scores(best_model_scores)
        print('===========================================================================')
        print("Doing final training and test")
        best_model_compile_params = {} if best_model_i >= len(self._model_compile_params) else \
            self._model_compile_params[best_model_i]

        best_model_builder = self._model_builders[best_model_i]
        train_data = TrainLoader(self._image_sizes[best_model_i]).load()
        test_data = TestLoader(self._image_sizes[best_model_i]).load()
        best_model, scores, history = self._compile_fit_model(
            best_model_builder, dict(best_model_compile_params), train_data, test_data)
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
            plt.savefig(f'cm_{best_model_i}.png')
            plt.show()
        best_model.save(f'model_{best_model_i}.h5')
        return best_model, best_model_i, scores, history

    @staticmethod
    def _compile_fit_model(model_builder, model_compile_params, train_data, validation_data):
        extra_intermediate_layers = []
        if "dense" in model_compile_params and model_compile_params["dense"] > 0:
            extra_intermediate_layers.append(Dense(model_compile_params["dense"], activation='relu'))
            del model_compile_params["dense"]
        if "dropout" in model_compile_params and model_compile_params["dropout"] > 0:
            extra_intermediate_layers.append(Dropout(model_compile_params["dropout"]))
            del model_compile_params["dropout"]

        model = model_builder(extra_intermediate_layers)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1,
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
        base_fit_params = {"epochs": 50, "validation_data": (validation_data.X, validation_data.y),
                           "callbacks": [es, learning_rate_reduction]}

        if type(train_data) is SupervisedData:
            extra_fit_params = {"x": train_data.X, "y": train_data.y}
        else:
            extra_fit_params = {"x": train_data}

        history = model.fit(**{**base_fit_params, **extra_fit_params})
        scores = model.evaluate(x=validation_data.X, y=validation_data.y, verbose=0)
        f1_score = 2 * scores[1] * scores[2] / (scores[1] + scores[2])
        return model, {"f1_score": f1_score, "accuracy": scores[0],
                "precision": scores[1], "recall": scores[2],
                "epochs": len(history.history['loss'])}, history

    @staticmethod
    def _report_scores(scores):
        print('F1 score: ', scores["f1_score"])
        print('Accuracy: ', scores["accuracy"])
        print('Precision: ', scores["precision"])
        print('Recall: ', scores["recall"])
        print('Epochs: ', scores["epochs"])
