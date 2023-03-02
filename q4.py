from scipy.io import wavfile
from sklearn.decomposition import FastICA
import numpy as np
import matplotlib.pyplot as plt


# Load wave files
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


# Plot amplitude, spectrogram, and frequency for each source
for i, source in enumerate(sources):
    # Create a new figure
    fig = plt.figure()

    # Amplitude vs. time plot
    time = np.arange(0, len(source)) / rates[i]
    plt.subplot(2, 1, 1)
    plt.plot(time, source)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f"Source {i+1}")

    # Spectrogram plot
    plt.subplot(2, 1, 2)
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(source, Fs=rates[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Show the figure for the current source
    plt.show()

# Make sure all the audio files have the same length
m = min([s.shape[0] for s in sources])
sources = [s[:m] for s in sources]

# Stack the sources into a matrix
S = np.c_[sources]


# Define the mixing matrix
mixing_matrix = np.random.uniform(0.5, 2.5, (6, 6))


# Generate observations
X = np.dot(S.T, mixing_matrix)


# Save the mixed signals as WAV files
for i in range(X.shape[1]):
    filename = f"audio/mixed_signal_{i+1}.wav"
    wavfile.write(filename, rates[i], X[:,i].astype(np.int16))


# Compute ICA
ica = FastICA()
S_ = ica.fit_transform(X)  # Get the estimated sources
A_ = ica.mixing_  # Get estimated mixing matrix

# Check if the original matrix can be reconstructed from the estimated sources and mixing matrix
np.allclose(X, np.dot(S_, A_.T))

# Compute mean and standard deviation for each separated signal
mean = np.mean(S_, axis=0)
std = np.std(S_, axis=0)

# Normalize each separated signal
S_normalized = (S_ - mean) / std

# Scale the outputs and write them to separate audio files
multiply_factor = 100
for i in range(S_normalized.shape[1]):
    temp_output = multiply_factor * S_normalized[:, i]
    wavfile.write(f"audio/Seperated_{i+1}.wav", rates[i], temp_output.astype(np.int16))

# Plot amplitude, spectrogram, and frequency for each separated signal
for i, S_normalized in enumerate(S_.T):
    # Create a new figure
    fig = plt.figure()

    # Amplitude vs. time plot
    time = np.arange(0, len(S_normalized)) / rates[i]
    plt.subplot(2, 1, 1)
    plt.plot(time, S_normalized)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f"Separated Signal {i+1}")

    # Spectrogram plot
    plt.subplot(2, 1, 2)
    powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(S_normalized, Fs=rates[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # Show the figure
    plt.show()




