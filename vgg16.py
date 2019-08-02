import tensorflow as tf
import numpy as np

class VGG16():
    model = None
    train_dataset = None
    train_labels = None
    test_dataset = None
    test_labels = None

    def __init__(self):
        super(VGG16, self).__init__()

        # Block 1
        self.model = tf.keras.models.Sequential()
        # Greyscale input images
        self.model.add(tf.keras.layers.ZeroPadding2D((1,1), input_shape=(224,224,3)))
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

    def load_dataset_labels(self, train=True, prefix=None):
        dataset = np.load(prefix+'dataset.npy') / 255
        labels = np.load(prefix + 'labels.npy')
        assert(np.shape(dataset)[0] == np.shape(labels)[0])
        if train:
            self.train_dataset = dataset
            self.train_labels = labels
        else:
            self.test_dataset = dataset
            self.test_labels = labels
        print('train' if train else 'test', dataset.shape)

    def train(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        self.model.fit(x=self.train_dataset, y=self.train_labels, epochs=5, steps_per_epoch=7)



if __name__ == "__main__":
    vgg = VGG16()
    print(vgg.model.summary())
    vgg.load_dataset_labels(train=True, prefix='Train-')
    vgg.load_dataset_labels(train=False, prefix='Test-')
    vgg.train()
    # print(tf.test.is_gpu_available())