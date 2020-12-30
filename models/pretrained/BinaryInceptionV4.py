from models.pretrained.InceptionV4 import InceptionV4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten


class BinaryInceptionV4(InceptionV4):
    def get_model(self):
        model = super().get_model()
        # flat = Flatten()(model.layers[-1].output)
        # classifier = Dense(128, activation='relu')(flat)
        # output = Dense(1, activation='sigmoid')(classifier)
        # Freeze the original layers weights
        # for index in range(len(model.layers) - 3):
        #   model.get_layer(index=index).trainable = False
        # return Model(inputs=model.inputs, outputs=output)
        return model
