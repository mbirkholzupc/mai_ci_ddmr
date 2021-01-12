# Using Neural Networks to Discriminate between Benign and Malignant Moles
## Computer Intelligence (CI) 2020 fall final project



### Project structure

* **data**
  * test - all: all the original test data as in *Kaggle*. Used for final model testing
  * train: all train (and validation data based off it)
    * all: all the original train data as in *Kaggle*. Used for final model training.
    * folds: 5-fold Strafied data used for cross-validation of the different models.
      * fold0 ... fold4: Each of the 5 folds numbered from 0 to 4.
* loader_splitter: classes related to the load and splitting of data.
* **models**: Neural network models implemented
  * core: Auxiliary classes for the training, validation and testing of models
    * test: Includes the runner to do the final training and test of the model with loss, F1-score and confussion matrix
    graphs as well as the final model in .h5 format with the whole architecture and with weights only.
    * validation: Includes the runner to get the best augmentation parameters as well as running multiple model 
      variations using the F1 score.  
  * **pretrained**: Includes the pretrained models, InceptionV4 and Mobilenet.
  * **scratch**: Includes the from-scratch model.  

### Scripts

The project contains several scripts in the root folder:

The Stratified 5-fold is generated using the `gen_cross_val_folds.py` script, we evaluate the best model (as desribed
above with `evaluate_best_model.py`) and then there is a single script for each network (and one for augmentation and
another without):

* `train_scratch_no_aug.py`: Model selection of the from-scratch network with **no** augmentation.
* `train_scratch_with_aug.py`: Model selection of the from-scratch network **with** augmentation.
* `train_mobilenet_no_aug.py`: Model selection of the MobileNet network with **no** augmentation.
* `train_mobilenet_with_aug.py`: Model selection of the MobileNet network **with** augmentation.
* `train_inception_no_aug.py`: Model selection of the Inception V4 network with **no** augmentation.
* `train_inception_with_aug.py`: Model selection of the Inception V4 network **with** augmentation.

*See our [webpage](https://skin-moles.web.app/) for additional resource like the paper and android app.*
