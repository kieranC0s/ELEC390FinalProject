# import h5py
#
# import pandas as pd
#
# import numpy as np
#
# from scipy import stats
#
# from sklearn.preprocessing import StandardScaler
#
#
# def data_filter(data, window_size):
#     filtered_data = []
#
#     for sample in data:
#
#         sample_data = []
#
#         for i in range(4):
#             window = sample[:, i]
#
#             # Remove outliers
#
#             z_scores = np.abs(stats.zscore(window))
#
#             window = window[z_scores < 3]
#
#             # Replace NaN values with the mean of the remaining values
#
#             window = pd.Series(window).astype(np.float64)
#
#             window[np.isnan(window)] = np.nanmean(window)
#
#             # Compute the simple moving average (SMA) of the data
#
#             window = window.rolling(window=window_size).mean()
#
#             # Normalize the data using the StandardScaler
#
#             scaler = StandardScaler()
#
#             window = scaler.fit_transform(window.values.reshape(-1, 1))
#
#             # Replace NaN values with linear interpolation
#
#             window = pd.Series(window.reshape(-1)).interpolate().values
#
#             sample_data.append(window)
#
#         # Pad the windows with zeros to ensure they have the same size
#
#         max_len = max([len(s) for s in sample_data])
#
#         padded_sample_data = [np.pad(s, (0, max_len - len(s))) for s in sample_data]
#
#         filtered_data.append(np.column_stack(padded_sample_data))
#
#     return np.array(filtered_data)
#
#
# def prepare_filtered_data(walking_data, jumping_data):
#     walking_data['label'] = 'walking'
#
#     jumping_data['label'] = 'jumping'
#
#     return pd.concat([walking_data, jumping_data], axis=0)
#
#
# import h5py
#
# # Open the HDF5 file in r+ mode
#
# f = h5py.File('data.h5', 'r+')
#
# # Extract the walking and jumping data from the 'Train' group
#
# train_walking_data = f['dataset/Train/Walking'][:]
#
# train_jumping_data = f['dataset/Train/Jumping'][:]
#
# # Extract the walking and jumping data from the 'Test' group
#
# test_walking_data = f['dataset/Test/Walking'][:]
#
# test_jumping_data = f['dataset/Test/Jumping'][:]
#
# # Filter the walking and jumping data using the data_filter function with a specified window size
#
# window_size = 5
#
# walking_filtered_train = data_filter(train_walking_data, window_size)
#
# jumping_filtered_train = data_filter(train_jumping_data, window_size)
#
# walking_filtered_test = data_filter(test_walking_data, window_size)
#
# jumping_filtered_test = data_filter(test_jumping_data, window_size)
#
# # Create new datasets with the filtered data
#
# dataset_group = f['dataset']
#
# training_group = dataset_group.create_group('Train_Filtered')
#
# training_group.create_dataset('walking_filtered', data=walking_filtered_train)
#
# training_group.create_dataset('jumping_filtered', data=jumping_filtered_train)
#
# testing_group = dataset_group.create_group('Test_Filtered')
#
# testing_group.create_dataset('walking_filtered', data=walking_filtered_test)
#
# testing_group.create_dataset('jumping_filtered', data=jumping_filtered_test)
#
# f.close()


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

def data_filter(data, window_size):
    filtered_data = []

    for sample in data:

        sample_data = []

        for i in range(4):
            window = sample[:, i]

            # Remove outliers
            z_scores = np.abs(stats.zscore(window))
            window = window[z_scores < 3]

            # Replace NaN values with the mean of the remaining values
            window = pd.Series(window).astype(np.float64)
            window[np.isnan(window)] = np.nanmean(window)

            # Compute the simple moving average (SMA) of the data
            window = window.rolling(window=window_size).mean()

            # Normalize the data using the StandardScaler
            scaler = StandardScaler()
            window = scaler.fit_transform(window.values.reshape(-1, 1))

            # Replace NaN values with linear interpolation
            window = pd.Series(window.reshape(-1)).interpolate().values

            sample_data.append(window)

        # Pad the windows with zeros to ensure they have the same size
        max_len = max([len(s) for s in sample_data])
        padded_sample_data = [np.pad(s, (0, max_len - len(s))) for s in sample_data]

        filtered_data.append(np.column_stack(padded_sample_data))

    return np.array(filtered_data)
f = h5py.File('data.h5', 'r')
# Load the walking and jumping data from CSV files
train_walking_data = pd.read_csv('dataset/Test/Jumping')
train_jumping_data = pd.read_csv('train_jumping_data.csv')
test_walking_data = pd.read_csv('test_walking_data.csv')
test_jumping_data = pd.read_csv('test_jumping_data.csv')

# Filter the walking and jumping data using the data_filter function with a specified window size
window_size = 5
walking_filtered_train = data_filter(train_walking_data, window_size)
jumping_filtered_train = data_filter(train_jumping_data, window_size)
walking_filtered_test = data_filter(test_walking_data, window_size)
jumping_filtered_test = data_filter(test_jumping_data, window_size)

# Save the filtered data to CSV files
np.savetxt("walking_filtered_train.csv", walking_filtered_train, delimiter=",")
np.savetxt("jumping_filtered_train.csv", jumping_filtered_train, delimiter=",")
np.savetxt("walking_filtered_test.csv", walking_filtered_test, delimiter=",")
np.savetxt("jumping_filtered_test.csv", jumping_filtered_test, delimiter=",")

def prepare_filtered_data(walking_data, jumping_data):
    walking_data['label'] = 'walking'
    jumping_data['label'] = 'jumping'
    return pd.concat([walking_data, jumping_data], axis=0)

f = h5py.File('data.h5', 'r')

# Load the walking and jumping data from CSV files
train_walking_data = f[f'dataset/Train/Walking']
train_jumping_data = f[f'dataset/Train/Jumping']
test_walking_data = f[f'dataset/Test/Walking']
test_jumping_data = f[f'dataset/Test/Jumping']

# Filter the walking and jumping data using the data_filter function with a specified window size
window_size = 5
walking_filtered_train = data_filter(train_walking_data, window_size)
jumping_filtered_train = data_filter(train_jumping_data, window_size)
walking_filtered_test = data_filter(test_walking_data, window_size)
jumping_filtered_test = data_filter(test_jumping_data, window_size)

filtered_train_df = prepare_filtered_data(pd.DataFrame(walking_filtered_train.reshape(-1, 4)),
                                    pd.DataFrame(jumping_filtered_train.reshape(-1, 4)))
filtered_train_df.to_csv(f'filtered_data_train.csv', index=False)

filtered_train_df = prepare_filtered_data(pd.DataFrame(walking_filtered_test.reshape(-1, 4)),
                                    pd.DataFrame(jumping_filtered_test.reshape(-1, 4)))
filtered_train_df.to_csv(f'filtered_data_test.csv', index=False)