import tensorflow as tf
import numpy as np
import sys

def load_data(file_path, num_classes, num_samples_per_class):
    data = np.load(file_path, allow_pickle=True)
    x, y = data["x"], data["y"]

    x = x.astype("float32")

    x_samples, y_samples = [], []
    classList = ['aircraft carrier', 'airplane', 'alarm clock']

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
    num_classes = 3
    num_samples_per_class = 1000  # Number of samples per class
    max_length = 400  

    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
        test_npz_file = sys.argv[2]
        print(f"Processing file: {npz_file} and {test_npz_file}")

        (x_train, y_train) = load_data(npz_file, num_classes, num_samples_per_class)
        x_train = x_train.reshape(-1, max_length, 3)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(max_length, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Custom learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=10000, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Compile the model
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        # Implement early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

        # Evaluate the model
        x_test, y_test = load_data(test_npz_file, num_classes, num_samples_per_class)
        x_test = x_test.reshape(-1, max_length, 3)

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
        print(f'Test accuracy: {test_accuracy}')

        # Save the trained model
        model.save('quickdraw_model.h5')


    else:
        print("No filename provided. Please provide a filename as a command line argument.")
        sys.exit(1)

if __name__ == "__main__":
    main()
