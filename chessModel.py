# Defines the actual model for making policy and value predictions.
from tensorflow.keras import Input
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Activation, Dense, Flatten
from tensorflow.python.keras.layers.merge import Add
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf
import sys


def create_uci_labels():
    """
    Creates the labels for the universal chess interface into an array and returns them
    :return:
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                [(l1, t) for t in range(8)] + \
                [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                [(l1 + a, n1 + b) for (a, b) in
                 [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + \
                        letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        l = letters[l1]
        for p in promoted_to:
            labels_array.append(l + '2' + l + '1' + p)
            labels_array.append(l + '7' + l + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(l + '2' + l_l + '1' + p)
                labels_array.append(l + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(l + '2' + l_r + '1' + p)
                labels_array.append(l + '7' + l_r + '8' + p)
    return labels_array


class ModelConfig:
    labels = create_uci_labels()

    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_first_filter_size = 5
        self.cnn_filter_size = 3
        self.res_layer_num = 7
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.distributed = False
        self.input_depth = 18
        self.n_labels = 1968
        self.labels = ModelConfig.labels


class ChessModel:
    """
    The model which can be trained to take observations of a game of chess
    and return value and policy predictions. Inpired by https://github.com/Zeta36/chess-alpha-zero/blob/master/src/chess_zero/agent/model_chess.py

    Attributes:
        :ivar Config config: configuration to use
        :ivar Model model: the Keras model to use for predictions
    """

    def __init__(self, config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None
        self.api = None

    def build(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        mc = self.config
        in_x = x = Input((12, 8, 8))

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size,
                   padding="same", data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)

        policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(
            mc.l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first",
                   use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(
            mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg),
                          activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x, index):
        # mc = self.config.model
        mc = self.config

        in_x = x
        res_name = "res"+str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size,
                   padding="same",
                   data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu", name=res_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size,
                   padding="same",
                   data_format="channels_first", use_bias=False,
                   kernel_regularizer=l2(mc.l2_reg),
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

    def compile(self, optimizer, loss, metrics, loss_weights=None):
        self.model.compile(optimizer=optimizer, loss=loss,
                           metrics=metrics, loss_weights=loss_weights)
        return self.model

    def fit(self, dataset, y=None, validation_data=None, batch_size=None, epochs=10,
            shuffle=True, val_split=None, callbacks=None):
        self.model.fit(x=dataset, y=y, batch_size=batch_size, epochs=epochs,
                       shuffle=True, validation_split=val_split,
                       validation_data=validation_data,
                       callbacks=callbacks)
        return self.model

    def predict(self, x, batch_size=None, steps=None, callbacks=None,
                max_queue_size=10, workers=1, use_multiprocessing=False):
        value = self.model.predict(x=x, batch_size=batch_size, steps=steps,
                                   callbacks=callbacks, max_queue_size=max_queue_size,
                                   workers=workers,
                                   use_multiprocessing=use_multiprocessing)
        return value

    def summary(self):
        self.model.summary(line_length=None, positions=None, print_fn=None)


# Config = ModelConfig()
# a = ChessModel(Config)
# a.build()
# a.model.summary()
# keras.utils.plot_model(
#     a.model, "my_first_model_with_shape_info.png", show_shapes=True)
