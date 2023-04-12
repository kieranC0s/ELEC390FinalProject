
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

def segment_data(data, window_size, overlap):
    segments = []
    for i in range(0, len(data) - window_size, overlap):
        segment = data[i:i + window_size]
        segments.append(segment)
    return segments

# Read in the CSV files for each person
kieran_walking = pd.read_csv('Accelerometer_Walking_Kieran.csv')
kieran_jumping = pd.read_csv('Accelerometer_Jumping_Kieran.csv')
amir_walking = pd.read_csv('Accelerometer_Walking_Amir.csv')
amir_jumping = pd.read_csv('Accelerometer_Jumping_Amir.csv')
jack_walking = pd.read_csv('Accelerometer_Walking_Jack.csv')
jack_jumping = pd.read_csv('Accelerometer_Jumping_Jack.csv')

# Define window size and overlap (assuming 100 samples per second)
window_size = 5 * 100
overlap = window_size // 2

# Segment data into 5-second windows with 50% overlap
kieran_walking_segments = segment_data(kieran_walking.values, window_size, overlap)
kieran_jumping_segments = segment_data(kieran_jumping.values, window_size, overlap)
amir_walking_segments = segment_data(amir_walking.values, window_size, overlap)
amir_jumping_segments = segment_data(amir_jumping.values, window_size, overlap)
jack_walking_segments = segment_data(jack_walking.values, window_size, overlap)
jack_jumping_segments = segment_data(jack_jumping.values, window_size, overlap)

# Create the HDF5 file
f = h5py.File('data.h5', 'w')

# Create group for each person
jack_group = f.create_group('Jack')
kieran_group = f.create_group('Kieran')
amir_group = f.create_group('Amir')

# Store jumping and walking segments under each person's group
jack_group.create_dataset('jumping', data=jack_jumping_segments)
jack_group.create_dataset('walking', data=jack_walking_segments)
kieran_group.create_dataset('jumping', data=kieran_jumping_segments)
kieran_group.create_dataset('walking', data=kieran_walking_segments)
amir_group.create_dataset('jumping', data=amir_jumping_segments)
amir_group.create_dataset('walking', data=amir_walking_segments)

# Create dataset group
dataset_group = f.create_group('dataset')

# Shuffle and store jumping and walking segments in Training and Testing groups under dataset group
jumping_segments = kieran_jumping_segments + amir_jumping_segments + jack_jumping_segments
walking_segments = kieran_walking_segments + amir_walking_segments + jack_walking_segments

shuffled_jumping_segments = train_test_split(jumping_segments, test_size=0.1, random_state=42)
shuffled_walking_segments = train_test_split(walking_segments, test_size=0.1, random_state=42)

# Store shuffled jumping and walking segments in Training group
training_group = dataset_group.create_group('Train')
training_group.create_dataset('Jumping', data=shuffled_jumping_segments[0])
training_group.create_dataset('Walking', data=shuffled_walking_segments[0])

# Store shuffled jumping and walking segments in Testing group
testing_group = dataset_group.create_group('Test')
testing_group.create_dataset('Jumping', data=shuffled_jumping_segments[1])
testing_group.create_dataset('Walking', data=shuffled_walking_segments[1])

# Close the HDF5 file
f.close()