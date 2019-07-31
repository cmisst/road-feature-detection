import tensorflow as tf

class VGG16():
    model = None

    def __init__(self):
        super(VGG16, self).__init__()

        # Block 1
        self.model = tf.keras.models.Sequential()
        # Greyscale input images
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1), input_shape=(224,224,1)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 2
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 3
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 4
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Block 5
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1)))
        self.model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # # Classification block
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(4096, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))


if __name__ == "__main__":
    vgg = VGG16()
    print(vgg.model.summary())
    print(tf.test.is_gpu_available())