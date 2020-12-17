import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Lambda, Cropping2D, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential


#  Use PIL.Image to read images in RGB format
def read_image(fname: str) -> np.ndarray:
    return np.array(Image.open(fname))


def main():
    # Seed RNG for repeatability
    np.random.seed(1)

    # Load data
    # It was found via experimentation that a generator was not needed to successfully train the model.
    # Hence, no generator is used.
    print("Loading data...")
    data_directory = "data"
    driving_log = pd.read_csv(os.path.join(data_directory, "driving_log.csv")).values
    X = []
    y = []
    steering_correction = 0.2  # steering correction for left and right camera images. Add correction for left images, subtract for right images
    for idx in range(driving_log.shape[0]):
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

    # Convert images and steering data to numpy arrays
    X_train = np.array(X)
    y_train = np.array(y)

    # validation split
    valid_split = 0.2

    # Preprocessing function, used as a Lambda layer at the beginning of the model
    def preprocessing(x):
        # Normalize each pixel to values in [-0.5, 0.5]
        return x/255.0 - 0.5

    # Construct model using Keras
    # Model is heavily based on suggested architecture from the lecture materials
    # Batch norm layers are added to speed up training while preventing overfitting
    model = Sequential()
    model.add(Lambda(preprocessing, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))  # Remove unnecessary pixels on top (sky) and bottom (frontal body of car) from each image
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("relu"))
    model.add(Dense(1))

    # Compile and fit the model
    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, y_train, epochs=5, validation_split=valid_split, shuffle=True, verbose=1)

    # Save model
    model_filename = "model.h5"
    model.save(model_filename)
    print("{} saved".format(model_filename))


if __name__ == "__main__":
    main()
