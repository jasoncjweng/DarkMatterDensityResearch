import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input
from keras.models import Model
from datetime import datetime

class DataLoader:
    def __init__(self, file_name, cube_size, num_rotations=19):
        self.file_name = file_name
        self.cube_size = cube_size
        self.num_rotations = num_rotations
        self.densities = None
        self.redfluxes = None
        self._load_data()

    # Load data from file, split into density and redshift-flux cubes
    def _load_data(self):
        data = np.fromfile(self.file_name, dtype=np.float32)
        density = data[1:256**3 + 1]
        redflux = data[(256**3) * 2 + 1:(256**3) * 3 + 1]
        self.density3d = np.reshape(density, (256, 256, 256))
        self.redflux3d = np.reshape(redflux, (256, 256, 256))
        self._augment_data()

    # Augment data by rotating across 90 degrees in all possible orientations + 1 mirroring
    def _augment_data(self):
        densities = np.empty((self.num_rotations, 256, 256, 256))
        redfluxes = np.empty((self.num_rotations, 256, 256, 256))
        densities[0], redfluxes[0] = self.density3d, self.redflux3d

        axes = [(0, 1), (1, 2), (2, 0)]
        j = 1
        for axis in axes:
            for i in range(1, 4):
                densities[j] = np.rot90(self.density3d, k=i, axes=axis)
                redfluxes[j] = np.rot90(self.redflux3d, k=i, axes=axis)
                j += 1

        self.density3d = np.flipud(self.density3d)
        self.redflux3d = np.flipud(self.redflux3d)

        for axis in axes:
            for i in range(1, 4):
                densities[j] = np.rot90(self.density3d, k=i, axes=axis)
                redfluxes[j] = np.rot90(self.redflux3d, k=i, axes=axis)
                j += 1

        self.densities, self.redfluxes = densities, redfluxes

    # Split data into smaller cubes for training and verification
    def prepare_data(self, train_num, test_num):
        size, cubes = self.cube_size, 256 // self.cube_size
        d_train, rf_train = np.ndarray((cubes**3 * train_num)), np.ndarray((cubes**3 * train_num, size, size, size))
        d_test, rf_test = np.ndarray((cubes**3 * test_num)), np.ndarray((cubes**3 * test_num, size, size, size))

        def fill_dataset(data, redflux, start, end):
            index = 0
            for h in range(start, end):
                for i in range(cubes):
                    for j in range(cubes):
                        for k in range(cubes):
                            data[index] = self.densities[h, int(size*(2*i+1)/2), int(size*(2*j+1)/2), int(size*(2*k+1)/2)]
                            redflux[index] = self.redfluxes[h, size*i:size*(i+1), size*j:size*(j+1), size*k:size*(k+1)]
                            index += 1

        fill_dataset(d_train, rf_train, 0, train_num)
        fill_dataset(d_test, rf_test, train_num, train_num + test_num)
        
        return np.reshape(d_train, (cubes**3 * train_num)), np.reshape(rf_train, (cubes**3 * train_num, size, size, size)), \
               np.reshape(d_test, (cubes**3 * test_num)), np.reshape(rf_test, (cubes**3 * test_num, size, size, size))


class Conv3DModel:
    def __init__(self, input_shape, dropout_rate=0.4):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=self.input_shape)
        x = Conv3D(8, kernel_size=(16, 16, 16), activation='tanh', padding='same',
                   kernel_regularizer=tf.keras.regularizers.L2(0.01))(input_layer)
        x = Conv3D(32, kernel_size=(16, 16, 16), activation='tanh', padding='same',
                   kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
        x = MaxPool3D(pool_size=(8, 8, 8), padding='same')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Flatten()(x)
        x = Dense(2048, activation='linear')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='linear')(x)
        x = Dropout(self.dropout_rate)(x)
        output_layer = Dense(1, activation='linear')(Dense(8, activation='linear')(x))

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])
        return model

    def train(self, rf_train, d_train, rf_test, d_test, batch_size=128, epochs=16):
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = [
            tf.keras.callbacks.CSVLogger('1pixel_history.csv', separator=",", append=True),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        return self.model.fit(rf_train, d_train, batch_size=batch_size, epochs=epochs, 
                              validation_data=(rf_test, d_test), callbacks=callbacks)

    def evaluate(self, rf_test, d_test, batch_size=64):
        return self.model.evaluate(rf_test, d_test, batch_size=batch_size)

    def save_model(self, file_path, n = None):
        if not n:
            name = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = f"models/1pixel/model_{name}.h5"
        self.model.save(file_path)
        print(f"Model saved as: {file_path}")


# Usage
data_loader = DataLoader('2.5_32_256.dat', cube_size=32)
d_train, rf_train, d_test, rf_test = data_loader.prepare_data(train_num=18, test_num=1)

conv3d_model = Conv3DModel(input_shape=(32, 32, 32, 1))
history = conv3d_model.train(rf_train, d_train, rf_test, d_test)
conv3d_model.save_model()

print("Evaluating model...")
results = conv3d_model.evaluate(rf_test, d_test)
print('Evaluation results:', results)
