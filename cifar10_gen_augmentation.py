import keras

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomTranslation(4/32, 4/32, fill_mode='nearest'),
])

data_augmentation.build(input_shape=(32, 32, 3))
data_augmentation.summary()

data_augmentation.save('./data/cifar10_augmentation.h5')