import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization



def load_data(file_path, num_classes, num_samples_per_class):
    data = np.load(file_path, allow_pickle=True)
    x, y = data["x"], data["y"]

    x = x.astype("float32")

    x_samples, y_samples = [], []
    # change to list of available types as needed
    classList = ['aircraft carrier', 'airplane', 'alarm clock']

    # create a unique identifier for each class
    class_to_int = {class_label: i for i, class_label in enumerate(classList)}

    min_samples_per_class = float('inf')  # Initialize with a high value

    for class_label in classList:
        class_indices = np.where(y == class_label)[0]

        # Determine the minimum number of samples per class
        min_samples_per_class = min(min_samples_per_class, len(class_indices))

    for class_label in classList:
        class_indices = np.where(y == class_label)[0]

        # Check if there are enough samples in this class
        if len(class_indices) >= num_samples_per_class:
            selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)
            x_samples.append(x[selected_indices])
            y_samples.append([class_to_int[class_label]] * num_samples_per_class)
        else:
            print(f"Warning: Class {class_label} has fewer than {num_samples_per_class} samples. Samples: {class_indices}")

    if not x_samples:
        raise ValueError("Not enough samples for any class. Check your dataset.")

    x_samples = np.concatenate(x_samples)
    y_samples = np.concatenate(y_samples)

    return x_samples, y_samples


def main():
    # Set these values
    num_classes = 3
    num_samples_per_class = 1000
    image_size = (50, 50)  # Modify to your desired image size

    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
        test_npz_file = sys.argv[2]
        print(f"Processing file: {npz_file} and {test_npz_file}")

        (x_train, y_train) = load_data(npz_file, num_classes, num_samples_per_class)

        # Reshape the input data to fit the convolutional model
        x_train = x_train.reshape(-1, *image_size, 1)
        # bring values between 0-1
        x_train = x_train / 255.0

        #min_samples = min(x_train.shape[0], y_train.shape[0])

        # Synchronize the data
        #x_train = x_train[:min_samples]
        #y_train = y_train[:min_samples]


        # Define a convolutional neural network model
        model = tf.keras.models.Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 1)),
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')

        ])

        # Reduce the learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # You can also use 0.0001

        # Compile the model with the modified optimizer
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, epochs=5)

        # Evaluate the model
        x_test, y_test = load_data(test_npz_file, num_classes, num_samples_per_class)
        x_test = x_test / 255.0
        x_test = x_test.reshape(-1, *image_size, 1)

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
        print(f'Test accuracy: {test_accuracy}')


        # Iterate through each image in the test dataset
        total_black_pixels = 0
        total_pixels = x_test.shape[1] * x_test.shape[2]

        for img in x_test:
            black_pixels = np.sum(img == 0)  # Assuming black is represented as 0
            total_black_pixels += black_pixels

        # Calculate the percentage of black pixels
        percent_black_pixels = (total_black_pixels / (len(x_test) * total_pixels)) * 100

        print(f"Percentage of black pixels in the test dataset: {percent_black_pixels:.2f}%")


        model.summary()

        # Save the trained model
        model.save('../models/quickdraw_model.h5')

    else:
        print("No filename provided. Please provide a filename as a command line argument.")
        sys.exit(1)


if __name__ == "__main__":
    main()