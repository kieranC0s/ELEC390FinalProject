import h5py
import numpy as np
from scipy import stats

def extract_features(data):
    features = []
    for window in data:
        feature_list = []
        for axis in range(1, 4):  # Iterate over x, y, and z axes
            axis_data = window[:, axis]

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
            feature_list.extend([max_value, min_value, range_value, mean_value, median_value,
                                 variance, skewness, rms, kurtosis, std_dev])

        features.append(feature_list)

    return np.array(features)

# Read the dataset using the h5py library
f = h5py.File('data.h5', 'r')

for name in ['Kieran', 'Amir', 'Jack']:
    # Extract the walking and jumping data
    walking_data = f[f'{name}/Train']['walking'][:]
    jumping_data = f[f'{name}/Train']['jumping'][:]
    test_data = f[f'{name}/Test']['walking'][:]

    # Extract the features from the walking, jumping, and test data
    walking_features = extract_features(walking_data)
    jumping_features = extract_features(jumping_data)
    test_features = extract_features(test_data)

    # Save the extracted features to CSV files
    np.savetxt(f'walking_features_{name}.csv', walking_features, delimiter=',')
    np.savetxt(f'jumping_features_{name}.csv', jumping_features, delimiter=',')
    np.savetxt(f'test_features_{name}.csv', test_features, delimiter=',')

f.close()