import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import scipy.sparse as sp
from keras import Model
import subprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_ALLOW_GROWTH'] = 'true'



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)



def _load_data_x_y(x_path: str,
                   y_path: str,
                   do_label_flip: bool
                   ) -> tf.data.Dataset:
    format = x_path.split('.')[-2:]
    if format[-1] == 'npy':
        x = np.load(x_path)
    elif format[-1] == 'npz' and format[-2] == 'csr':
        x = sp.load_npz(x_path).toarray()
    else:
        raise ValueError(f'Unknown format: {format}')

    y = np.load(y_path)
    y_max = np.max(y)
    y_min = np.min(y)

    if do_label_flip:
        y = y_max - y + y_min

    y = y.astype(np.float32) / 4.0

    ret = tf.data.Dataset.from_tensor_slices((x, y))

    return ret



# init_model_path = './data/sentiment140_init_model.h5'
x_train_path = './data/sentiment140_x_train_9.csr.npz'
y_train_path = './data/sentiment140_y_train_9.npy'
x_test_path = './data/sentiment140_x_train_1.csr.npz'
y_test_path = './data/sentiment140_y_train_1.npy'
batch_size = 1024
shuffle_train = True
repeat_train = False

_SEQUENCE_LENGTH = 60
_LSTM_SIZE = 128
_KEY_DIM = 256
_WEIGHT_DECAY = 1e-3
_ATTENTION_DROPOUT = 0.6
_DENSE_DROPOUT = 0.4
_EPOCHS = 80



embedding_matrix = np.load('./data/sentiment140_embedding.npy')
# not-exists word

embedding_matrix = np.pad(embedding_matrix, ((0,0),(0,1)), 'constant', constant_values=0)
embedding_matrix[-1][-1] = 1



tf.keras.backend.clear_session()



inputs = tf.keras.layers.Input(shape=(_SEQUENCE_LENGTH,))
x = tf.keras.layers.Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False,mask_zero=True)(inputs)
# test BN
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(_LSTM_SIZE, return_sequences=True))(x)
# x = tf.keras.layers.LSTM(_LSTM_SIZE, return_sequences=True)(x)

x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=_KEY_DIM, dropout=_ATTENTION_DROPOUT)(x, x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(_DENSE_DROPOUT)(x)
x = tf.keras.layers.BatchNormalization()(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='sentiment140_init_model')
model.summary()



# model: Model = tf.keras.models.load_model(init_model_path, compile=True)
# model.summary()



with tf.device('/cpu:0'):
    train_ds = _load_data_x_y(x_train_path, y_train_path, do_label_flip=False)
    test_ds = _load_data_x_y(x_test_path, y_test_path, do_label_flip=False)
    print("length of train_ds:", len(train_ds))
    print("length of test_ds:", len(test_ds))
    # val_ds = None
    # TODO: validation dataset may NOT be `None`

    # all_ds = _load_data_x_y(x_train_path, y_train_path, do_label_flip=False)
    # test_ds = all_ds.take(len(all_ds) // 6)
    # train_ds = all_ds.skip(len(all_ds) // 6)

    steps_per_epoch = (len(train_ds) + batch_size - 1) // batch_size

    if shuffle_train:
        train_ds = train_ds.shuffle(buffer_size=len(train_ds))

    if repeat_train:
        train_ds = train_ds.repeat()

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-3,
#     decay_steps=steps_per_epoch*3,
#     decay_rate=0.3,
#     staircase=True
# )

lr_schedule = 1e-3



model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tfa.optimizers.AdamW(weight_decay=_WEIGHT_DECAY, learning_rate=lr_schedule),
    # optimizer=tf.keras.optimizers.Adam(lr_schedule),
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
)



history = model.fit(
    train_ds,
    epochs=_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_freq=1,
    callbacks=[tf.keras.callbacks.CSVLogger('./logs/sentiment140_train.log')]
)





