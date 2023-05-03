import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import pydot
import tensorflow_datasets as tfds
# DenseNet
def densenet(input_shape, num_classes, growth_rate=32, depth=40, compression_rate=0.5):
    n_blocks = (depth - 4) // 6
    n_channels = growth_rate * 2

    def dense_block(x, n_blocks):
        for i in range(n_blocks):
            x1 = layers.BatchNormalization()(x)
            x1 = layers.Activation('relu')(x1)
            x1 = layers.Conv2D(4 * growth_rate, (1, 1), padding='same', kernel_initializer='he_normal')(x1)
            x1 = layers.BatchNormalization()(x1)
            x1 = layers.Activation('relu')(x1)
            x1 = layers.Conv2D(growth_rate, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
            x = layers.Concatenate(axis=-1)([x, x1])
        return x

    def transition_layer(x, compression_rate):
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(int(tf.keras.backend.int_shape(x)[-1] * compression_rate), (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = layers.AveragePooling2D((2, 2), strides=2)(x)
        return x

    inputs = Input(shape=input_shape)
    x = layers.Conv2D(n_channels, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = dense_block(x, n_blocks)
    x = transition_layer(x, compression_rate)
    x = dense_block(x, n_blocks)
    x = transition_layer(x, compression_rate)
    x = dense_block(x, n_blocks)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

# Yaratish va kompilyatsiya
input_shape = (32, 32, 3)
num_classes = 12
batch_size = 128
def preprocess_data(example):
    image = example['image']
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 255.0
    label = example['super_class_id']
    return image, label

dataset = tfds.load("stanford_online_products", split = "train")

ds_test = tfds.load("stanford_online_products", split = "test")
# Preprocess the data


dataset = dataset.map(preprocess_data)
ds_test = ds_test.map(preprocess_data)

dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

model = densenet(input_shape, num_classes)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
batch_size = 128
epochs = 1
plot_model(model, to_file='densenet.png', show_shapes=True)
history = model.fit(dataset, epochs=epochs, validation_data=ds_test)
