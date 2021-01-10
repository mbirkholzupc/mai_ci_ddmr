from models.pretrained.inceptionv4.base.InceptionV4 import InceptionV4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten


class BinaryInceptionV4(InceptionV4):
    def get_model(self, *argv):
        model = super().get_model()

        # Freeze the original layers weights
        for index in range(len(model.layers) - 22):
            model.get_layer(index=index).trainable = False

        x = Flatten()(model.layers[-1].output)
        intermediate_layers = argv[0]
        for layer in intermediate_layers:
            x = layer(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=model.inputs, outputs=output)

