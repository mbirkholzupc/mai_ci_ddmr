from models.pretrained.inceptionv4.base.InceptionV4 import InceptionV4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten


class BinaryInceptionV4(InceptionV4):
    def get_model(self, *argv):
        model = super().get_model()
        x = Flatten()(model.layers[-1].output)
        intermediate_layers = argv[0]
        for layer in intermediate_layers:
            x = layer(x)
        output = Dense(1, activation='sigmoid')(x)
        # Freeze the original layers weights
        for index in range(len(model.layers) - len(intermediate_layers)):
            model.get_layer(index=index).trainable = False
        return Model(inputs=model.inputs, outputs=output)
