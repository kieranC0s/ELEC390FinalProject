import h5py
import pandas as pd

# Read the dataset using the h5py library
with h5py.File('data.h5', 'r') as f:
    # Extract the test data for each person
    kieran_test_walking = f['Kieran/Test']['walking'][:]
    kieran_test_jumping = f['Kieran/Test']['jumping'][:]
    amir_test_walking = f['Amir/Test']['walking'][:]
    amir_test_jumping = f['Amir/Test']['jumping'][:]
    jack_test_walking = f['Jack/Test']['walking'][:]
    jack_test_jumping = f['Jack/Test']['jumping'][:]

# Convert the test data to DataFrames
kieran_test_walking_df = pd.DataFrame(kieran_test_walking.reshape(-1, 4))
kieran_test_jumping_df = pd.DataFrame(kieran_test_jumping.reshape(-1, 4))
amir_test_walking_df = pd.DataFrame(amir_test_walking.reshape(-1, 4))
amir_test_jumping_df = pd.DataFrame(amir_test_jumping.reshape(-1, 4))
jack_test_walking_df = pd.DataFrame(jack_test_walking.reshape(-1, 4))
jack_test_jumping_df = pd.DataFrame(jack_test_jumping.reshape(-1, 4))

# Save the test data as CSV files
kieran_test_walking_df.to_csv('kieran_test_walking.csv', index=False)
kieran_test_jumping_df.to_csv('kieran_test_jumping.csv', index=False)
amir_test_walking_df.to_csv('amir_test_walking.csv', index=False)
amir_test_jumping_df.to_csv('amir_test_jumping.csv', index=False)
jack_test_walking_df.to_csv('jack_test_walking.csv', index=False)
jack_test_jumping_df.to_csv('jack_test_jumping.csv', index=False)
