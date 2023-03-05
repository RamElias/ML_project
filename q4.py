"""
================================
PROJECT - Question 4 - Machine learning
================================

Name: Eliezer Seror  Id: 312564776
Name: Ram Elias      Id: 205445794

"""

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from scipy.io import wavfile
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


# Plot amplitude, spectrogram, and frequency for each separated signal
def plot_results(time, source, rate, i):
    plt.subplot(2, 1, 1)
    plt.plot(time, source)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f"Source {i + 1}")

    # Spectrogram plot
    plt.subplot(2, 1, 2)
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(source, Fs=rate)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.show()


# load 6 sources, create new 6 mixed sources by a random matrix, use fastICA to recreate the
# original sources.
def ComputeICA():
    # Load wave sources
    sources = []
    rates = []
    for i in range(1, 7):
        filename = f"audio/source{i}.wav"
        rate, data = wavfile.read(filename)
        sources.append(data)
        rates.append(rate)
        length = len(data)
        print(f"Data for {filename}: {data}")
        print(f"Length of {filename}: {length}")

    # Plot each source
    for i, source in enumerate(sources):
        time = np.arange(0, len(source)) / rates[i]
        plot_results(time, source, rates[i], i)

    # Make sure all the audio files have the same length
    m = min([s.shape[0] for s in sources])
    sources = [s[:m] for s in sources]

    # Stack the sources to a matrix
    S = np.c_[sources]

    # mixing matrix
    mixing_matrix = np.random.uniform(0.5, 2.5, (6, 6))

    # Generate observations
    X = np.dot(S.T, mixing_matrix)

    # Save the result of the mixed signals
    for i in range(X.shape[1]):
        filename = f"audio/mixed_signal_{i + 1}.wav"
        wavfile.write(filename, rates[i], X[:, i].astype(np.int16))

    # Compute ICA
    ica = FastICA()
    S_ = ica.fit_transform(X)
    A_ = ica.mixing_

    # Check if the original matrix can be reconstructed from the estimated sources and mixing matrix
    np.allclose(X, np.dot(S_, A_.T))

    # Compute mean and standard deviation for each separated signal
    mean = np.mean(S_, axis=0)
    std = np.std(S_, axis=0)

    # Normalize each separated signal
    S_normalized = (S_ - mean) / std

    # Scale the outputs and save the results
    multiply_factor = 100
    for i in range(S_normalized.shape[1]):
        temp_output = multiply_factor * S_normalized[:, i]
        wavfile.write(f"audio/seperated_{i + 1}.wav", rates[i], temp_output.astype(np.int16))

    # Plot each source
    for i, S_normalized in enumerate(S_.T):
        time = np.arange(0, len(S_normalized)) / rates[i]
        plot_results(time, S_normalized, rates[i], i)

    # loud 6 sources and split each one of them to 500 parts


# create a label arry using to_categorical for each part by his origin index
def load_data(source_files):
    sources = []
    labels = []
    for i, file in enumerate(source_files):
        data, sr = librosa.load(file, sr=None, mono=True, dtype=np.float32)
        parts = np.array_split(data, 500)
        sources.append(parts)
        labels.append(np.ones(len(parts)) * i)
    sources = np.concatenate(sources, axis=0)
    labels = np.concatenate(labels)
    labels = to_categorical(labels, num_classes=len(source_files))
    mean = np.mean(sources)
    std = np.std(sources)
    sources = (sources - mean) / std
    return sources, labels


def preprocess_data(sources):
    sources = np.expand_dims(sources, axis=2)
    return sources


# build and define the classifier with all the feacher needed
# set an optimizer (Adam)
# compile the model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    opt = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# train the model
def train_model(model, train_data, train_labels, val_split=0.2, epochs=500, verbose=2):
    history = model.fit(train_data, train_labels, validation_split=val_split, epochs=epochs, verbose=verbose)
    return history


# evaluate the model
def evaluate_model(model, test_data, test_labels):
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
    return ("Test accuracy:", test_acc)


# load data and split it, build and train a model
# use the model on the data and evaluate the result
def source_recognition(source_files):
    # Load data
    sources, labels = load_data(source_files)

    # Split data into training and testing sets
    train_sources, test_sources, train_labels, test_labels = train_test_split(sources, labels, test_size=0.2,
                                                                              random_state=42)

    # Preprocess data
    train_sources = preprocess_data(train_sources)
    test_sources = preprocess_data(test_sources)

    # Build and train model
    model = build_model(train_sources.shape[1:], len(source_files))
    history = train_model(model, train_sources, train_labels)

    # Evaluate model on test set
    return evaluate_model(model, test_sources, test_labels)


if __name__ == "__main__":
    # separate mix audio to there origin
    ComputeICA()

    # source_recognition for original sources
    source_files = ['audio/source1.wav', 'audio/source2.wav', 'audio/source3.wav', 'audio/source4.wav',
                    'audio/source5.wav', 'audio/source6.wav']
    original_source = source_recognition(source_files)

    # source_recognition for Seperated sources
    source_files = ['audio/seperated_1.wav', 'audio/seperated_2.wav', 'audio/seperated_3.wav', 'audio/seperated_4.wav',
                    'audio/seperated_5.wav', 'audio/seperated_6.wav']
    Seperated_source = source_recognition(source_files)

    # print the results
    print("\noriginal sources \n", original_source, "\n")
    print("Seperated sources \n", Seperated_source)
