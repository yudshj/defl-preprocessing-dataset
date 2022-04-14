import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3200)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


def load_data_x_y(x_path: str, y_path: str, do_label_flip: bool, to_one_hot: bool, normalize: bool = True):
    x = np.load(x_path)
    y = np.load(y_path)
    y_max = np.max(y)
    y_min = np.min(y)

    if do_label_flip:
        y = y_max - y + y_min
    if to_one_hot:
        y = tf.keras.utils.to_categorical(y, num_classes=y_max - y_min + 1)

    if normalize:
        x = x.astype(np.float32) / 255.

    return tf.data.Dataset.from_tensor_slices((x, y))



# tf.keras.backend.clear_session()


# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip('horizontal'),
#     tf.keras.layers.RandomTranslation(4/32, 4/32, fill_mode='nearest'),
# ])

data_augmentation = tf.keras.models.load_model('./data/cifar10_augmentation.h5')


with tf.device('/cpu:0'):
    ds = load_data_x_y('./data/cifar10_x_train.npy', './data/cifar10_y_train.npy', False, True, True).map(lambda img,label: (data_augmentation(img),label), num_parallel_calls=8)
    ds = ds.shuffle(buffer_size=50000).batch(32).prefetch(tf.data.AUTOTUNE)
    # ds = ds.repeat()
    test_ds = load_data_x_y('./data/cifar10_x_test.npy', './data/cifar10_y_test.npy', False, True, True)
    test_ds = test_ds.batch(128).prefetch(tf.data.AUTOTUNE)



model = tf.keras.models.load_model('./data/cifar10_init_model_big.h5')
model.summary()



history = model.fit(
    ds,
    epochs=80,
    # steps_per_epoch=800,
    validation_data=test_ds,
    validation_freq=1,
    callbacks=[tf.keras.callbacks.CSVLogger('./logs/cifar10_train_big.log')]
)

print(history.history['val_accuracy'])

