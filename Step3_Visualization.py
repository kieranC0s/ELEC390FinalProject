import h5py
import matplotlib.pyplot as plt

# Read segmented data from HDF5 file
with h5py.File('data.h5', 'r') as f:
    kieran_walking_segments = f['Kieran/walking'][:]
    kieran_jumping_segments = f['Kieran/jumping'][:]
    amir_walking_segments = f['Amir/walking'][:]
    amir_jumping_segments = f['Amir/jumping'][:]
    jack_walking_segments = f['Jack/walking'][:]
    jack_jumping_segments = f['Jack/jumping'][:]


# Function to plot acceleration vs. time for a sample
def plot_sample(sample, title):
    plt.figure(figsize=(10, 5))
    plt.plot(sample[:, 0], label='x-axis')
    plt.plot(sample[:, 1], label='y-axis')
    plt.plot(sample[:, 2], label='z-axis')
    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()


# Function to plot histogram of acceleration data
def plot_histogram(data, title, axis_labels=['x-axis', 'y-axis', 'z-axis'], bins=50):
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(axis_labels):
        plt.hist(data[:, i], bins=bins, alpha=0.5, label=label)
    plt.title(title)
    plt.xlabel('Acceleration')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# Plot a few samples and histograms for each person and for both classes (walking and jumping)
for i in range(3):
    plot_sample(kieran_walking_segments[i], f'Kieran - Walking Sample {i+1}')
    plot_sample(kieran_jumping_segments[i], f'Kieran - Jumping Sample {i+1}')
    plot_histogram(kieran_walking_segments[i], f'Kieran - Walking Histogram {i+1}')
    plot_histogram(kieran_jumping_segments[i], f'Kieran - Jumping Histogram {i+1}')

    plot_sample(amir_walking_segments[i], f'Amir - Walking Sample {i+1}')
    plot_sample(amir_jumping_segments[i], f'Amir - Jumping Sample {i+1}')
    plot_histogram(amir_walking_segments[i], f'Amir - Walking Histogram {i+1}')
    plot_histogram(amir_jumping_segments[i], f'Amir - Jumping Histogram {i+1}')

    plot_sample(jack_walking_segments[i], f'Jack - Walking Sample {i+1}')
    plot_sample(jack_jumping_segments[i], f'Jack - Jumping Sample {i+1}')
    plot_histogram(jack_walking_segments[i], f'Jack - Walking Histogram {i+1}')
    plot_histogram(jack_jumping_segments[i], f'Jack - Jumping Histogram {i+1}')