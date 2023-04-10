import h5py
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


def prepare_filtered_data(walking_data, jumping_data):
    walking_data['label'] = 'walking'
    jumping_data['label'] = 'jumping'
    return pd.concat([walking_data, jumping_data], axis=0)


# Read the dataset using the h5py library
f = h5py.File('data.h5', 'r')

for name in ['Kieran', 'Amir', 'Jack']:
    # Extract the walking and jumping data from the 'Train' group
    walking_data = f[f'{name}/Train']['walking'][:]
    jumping_data = f[f'{name}/Train']['jumping'][:]

    # Filter the walking and jumping data using the data_filter function with a specified window size
    window_size = 5
    walking_filtered = data_filter(walking_data, window_size)
    jumping_filtered = data_filter(jumping_data, window_size)

    # Save the filtered data to CSV files
    filtered_df = prepare_filtered_data(pd.DataFrame(walking_filtered.reshape(-1, 4)),
                                        pd.DataFrame(jumping_filtered.reshape(-1, 4)))
    filtered_df.to_csv(f'filtered_data_{name}.csv', index=False)

f.close()