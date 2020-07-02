import numpy as np
import tensorflow as tf
import chessModel
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from datetime import datetime
from tensorflow import keras
from pathlib import WindowsPath

# Input paths
inputPath = WindowsPath('C:/Pool')

with open('data.npy', 'rb') as f:
    stateList3 = np.load(f)
    moveList3 = np.load(f)
    resultList3 = np.load(f)

# Create model
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
config = chessModel.ModelConfig()
model = chessModel.ChessModel(config)
model.build()

opt = tf.keras.optimizers.Adam()
losses = ['categorical_crossentropy', 'mean_squared_error']
model.compile(optimizer=opt, loss=losses, metrics=["mae"],
              loss_weights=[0.1, 0.9])

logs = inputPath / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logs),
                                                      histogram_freq=1,
                                                      profile_batch='100,110')

kala = model.fit(dataset=stateList3, y=[moveList3, resultList3], batch_size=1024,
                 epochs=20, shuffle=True, val_split=0.2, validation_data=None,
                 callbacks=[tensorboard_callback])