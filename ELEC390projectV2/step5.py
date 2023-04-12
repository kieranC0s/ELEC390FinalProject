import h5py
import numpy as np
from scipy import stats


def extract_features(data, label):
    features = np.zeros((data.shape[0], 10, 3))
    for i in range(data.shape[0]):
        window = data[i]
        for axis in range(1, 4):  # Iterate over x, y, and z axes
            axis_data = window[:, axis - 1]

            if np.isnan(axis_data).any():
                print(f"Warning: NaN values found in axis_data at index {i}, label {label}")
                continue

            # Calculate the features for each axis
            max_value = np.max(axis_data)
            min_value = np.min(axis_data)
            range_value = max_value - min_value
            mean_value = np.mean(axis_data)
            median_value = np.median(axis_data)
            variance = np.var(axis_data)
            skewness = stats.skew(axis_data)
            rms = np.sqrt(np.mean(axis_data ** 2))
            kurtosis = stats.kurtosis(axis_data)
            std_dev = np.std(axis_data)

            # Add the features to the feature list
            features[i, :, axis - 1] = [max_value, min_value, range_value, mean_value, median_value,
                                        variance, skewness, rms, kurtosis, std_dev]

    # Combine features from all axes
    all_features = np.concatenate((features[:, :, 0], features[:, :, 1], features[:, :, 2]), axis=0)
    all_labels = np.full((all_features.shape[0], 1), label)
    all_features = np.hstack((all_features, all_labels))

    return all_features


# Read the dataset using the h5py library
f = h5py.File('data.h5', 'r')

# Extract the walking and jumping data
walking_data_train = f[f'dataset/Train_Filtered']['walking_filtered']
jumping_data_train = f[f'dataset/Train_Filtered/jumping_filtered'][:]
walking_data_test = f[f'dataset/Test_Filtered/walking_filtered'][:]
jumping_data_test = f[f'dataset/Test_Filtered/jumping_filtered'][:]

# Check for NaN values in the data
if np.isnan(walking_data_train).any() or np.isnan(jumping_data_train).any() or \
        np.isnan(walking_data_test).any() or np.isnan(jumping_data_test).any():
    print("Warning: NaN values found in the input data")

# Extract the features from the walking, jumping, and test data with labels
train_walking_features = extract_features(walking_data_train, 0)
train_jumping_features = extract_features(jumping_data_train, 1)
test_walking_features = extract_features(walking_data_test, 0)
test_jumping_features = extract_features(jumping_data_test, 1)

# Combine both walking and jumping data for training and testing
train_features = np.concatenate((train_walking_features, train_jumping_features), axis=0)
test_features = np.concatenate((test_walking_features, test_jumping_features), axis=0)

# Save the extracted features to CSV files
np.savetxt(f'train_features.csv', train_features, delimiter=',')
np.savetxt(f'test_features.csv', test_features, delimiter=',')
f.close()