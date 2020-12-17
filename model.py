from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import cv2
import time
from PIL import Image


def read_image(fname: str) -> np.ndarray:
    return np.array(Image.open(fname))


def main():
    # Seed RNG
    np.random.seed(1)

    # Load data
    print("Loading data...")

    data_directory = os.path.join("..", "data")
    driving_log = pd.read_csv(os.path.join(data_directory, "driving_log.csv")).values
    X = []
    y = []
    steering_correction = 0.2
    for idx in range(driving_log.shape[0]):
        try:
            steering = driving_log[idx, 3]
            # Center
            X.append(read_image(os.path.join(data_directory, driving_log[idx, 0].strip())))
            y.append(steering)
            # Left
            X.append(read_image(os.path.join(data_directory, driving_log[idx, 1].strip())))
            y.append(steering + steering_correction)
            # Right
            X.append(read_image(os.path.join(data_directory, driving_log[idx, 2].strip())))
            y.append(steering - steering_correction)
        except:
            print("Failed to load row {}".format(idx))

    # print("Augmenting data...")
    # # Augment via horizontal flip, negate steering
    # original_data_length = len(X)
    # for i in range(original_data_length):
    #     X.append(cv2.flip(X[i], 1))  # flip horizontally
    #     y.append(-1.0 * y[i])

    # Convert images and steering data to numpy arrays
    X_train = np.array(X)
    y_train = np.array(y)

    # Train/validation/test split
    # train_split = 0.99
    valid_split = 0.2

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_split, random_state=0)

    # Preprocessing function
    def preprocessing(x):
        return x / 255.0 - 0.5

    # Build the Final Test Neural Network in Keras Here
    model = Sequential()
    model.add(Lambda(preprocessing, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(1))

    # compile and fit the model
    model.compile(loss="mse", optimizer="adam")

    history = model.fit(X_train, y_train, epochs=5, validation_split=valid_split, shuffle=True, verbose=1)

    # Save model
    model_filename = "model.h5"
    model.save(model_filename)
    print("{} saved".format(model_filename))

    # Test model
    # metrics = model.evaluate(X_test, y_test)
    # for metric_i in range(len(model.metrics_names)):
    #     metric_name = model.metrics_names[metric_i]
    #     metric_value = metrics[metric_i]
    #     print('{}: {}'.format(metric_name, metric_value))


if __name__ == "__main__":
    main()
