import tensorflow as tf
from datetime import datetime
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import chessModel
import sequencer

s = sequencer.SimpleFeeder(['data1.npy', 'data2.npy', 'data3.npy', 'data4.npy', 'data5.npy'], files_per_batch=2, batch_size=1024)
train = s.get_train()
val = s.get_validation()

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

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                      histogram_freq=1,
                                                      profile_batch='100,110')

kala = model.fit(dataset=train,
                 validation_data=val,
                 epochs=2,
                 shuffle=False,
                 callbacks=[tensorboard_callback])
