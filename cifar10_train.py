import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import socket

if socket.gethostname() == 'GPU-SERVER':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 3GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

_NUM_CLASSES = 10
_LEARNING_RATE = 1e-3


def load_data_x_y(x_path: str, y_path: str, do_label_flip: bool, to_one_hot: bool, normalize: bool = True):
    x = np.load(x_path)
    y = np.load(y_path)

    if do_label_flip:
        y = _NUM_CLASSES - y - 1
    if to_one_hot:
        y = tf.one_hot(y, depth=_NUM_CLASSES)
        # y = tf.keras.utils.to_categorical(y, num_classes=_NUM_CLASSES)

    if normalize:
        x = x.astype(np.float32) / 255.

    return tf.data.Dataset.from_tensor_slices((x, y))


# tf.keras.backend.clear_session()


# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip('horizontal'),
#     tf.keras.layers.RandomTranslation(4/32, 4/32, fill_mode='nearest'),
# ])


with tf.device('/cpu:0'):
    data_augmentation = tf.keras.models.load_model('./data/cifar10_augmentation.h5')
    ds = load_data_x_y('./data/cifar10_x_train.npy', './data/cifar10_y_train.npy', False, True, True).map(
        lambda img, label: (data_augmentation(img), label), num_parallel_calls=8)
    ds = ds.shuffle(buffer_size=50000).batch(32).prefetch(tf.data.AUTOTUNE)
    # ds = ds.repeat()
    test_ds = load_data_x_y('./data/cifar10_x_test.npy', './data/cifar10_y_test.npy', False, True, True)
    test_ds = test_ds.batch(128).prefetch(tf.data.AUTOTUNE)


model = tf.keras.models.load_model('./data/cifar10_init_model.h5')
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(_LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[tf.metrics.CategoricalAccuracy()],
)
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./model/cifar10_model_{epoch:03d}.h5',
                                       monitor='val_categorical_accuracy',
                                       save_best_only=True),
    tf.keras.callbacks.CSVLogger('./logs/cifar10_log.csv'),
]

history = model.fit(
    ds,
    epochs=80,
    # steps_per_epoch=800,
    validation_data=test_ds,
    validation_freq=1,
    callbacks=[tf.keras.callbacks.CSVLogger('./logs/cifar10_train.log')]
)

print(history.history['val_accuracy'])
