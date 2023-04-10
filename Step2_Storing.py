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

# Create the HDF5 file and base group
f = h5py.File('data.h5', 'w')

# Create groups for each person and add segmented walking and jumping datasets
for name, walking_segments, jumping_segments in [('Kieran', kieran_walking_segments, kieran_jumping_segments),
                                                 ('Amir', amir_walking_segments, amir_jumping_segments),
                                                 ('Jack', jack_walking_segments, jack_jumping_segments)]:
    person_group = f.create_group(name)
    person_group.create_dataset('walking', data=walking_segments)
    person_group.create_dataset('jumping', data=jumping_segments)

    # Shuffle and split walking and jumping data into train and test sets
    walking_train, walking_test = train_test_split(walking_segments, test_size=0.1, random_state=42)
    jumping_train, jumping_test = train_test_split(jumping_segments, test_size=0.1, random_state=42)

    # Create 'Train' group and add walking and jumping train datasets
    train_group = f.create_group(f'{name}/Train')
    train_group.create_dataset('walking', data=walking_train)
    train_group.create_dataset('jumping', data=jumping_train)

    # Create 'Test' group and add walking and jumping test datasets
    test_group = f.create_group(f'{name}/Test')
    test_group.create_dataset('walking', data=walking_test)
    test_group.create_dataset('jumping', data=jumping_test)

# Close the HDF5 file
f.close()