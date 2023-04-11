import numpy as np
import pandas as pd
from scipy import stats
from joblib import load
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import ttk

def feature_extraction(data, window_size):
    filtered_data = []
    for i in range(0, len(data) - window_size, window_size):
        window = data[i:i + window_size]
        feature_list = []
        for axis in range(1, 4):  # Iterate over x, y, and z axes
            axis_data = window[:, axis]

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

            feature_list.extend([max_value, min_value, range_value, mean_value, median_value,
                                 variance, skewness, rms, kurtosis, std_dev])

        filtered_data.append(feature_list)

    return np.array(filtered_data)

def acquire_path():
    file_path = filedialog.askopenfilename()
    return file_path

def acquire_name():
    file_name = simpledialog.askstring("Output File", "Enter output CSV file name:")
    return file_name

def process_data():
    # Acquire input file path and output file name
    input_file_path = acquire_path()
    output_file_name = acquire_name()

    # Read input data file
    input_data = pd.read_csv(input_file_path).values

    # Extract features from input data
    window_size = 5 * 100  # Assuming 100 samples per second
    features = feature_extraction(input_data, window_size)

    # Load the trained logistic regression classifier
    classifier_name = 'classifier_Kieran.joblib'  # Choose the classifier you want to use
    classifier = load(classifier_name)

    # Make predictions on the feature array using the classifier
    predictions = classifier.predict(features)

    # Create a pandas DataFrame with the predicted activity labels
    output_data = pd.DataFrame({'activity': predictions})
    output_data['activity'] = output_data['activity'].replace({0.0: 'walking', 1.0: 'jumping'})

    # Save the output data to a CSV file with the desired file name
    output_data.to_csv(output_file_name, index=False)

    result_label.config(text=f'Result saved to: {output_file_name}')

# Create the main window
root = tk.Tk()
root.title("Activity Recognition")
root.geometry("350x200")

# Create labels, button, and result label
welcome_label = ttk.Label(root, text="Welcome to Activity Recognition!", font=("Helvetica", 14))
instruction_label = ttk.Label(root, text="Select a data file and enter a name for the output file.")
process_button = ttk.Button(root, text="Process Data", command=process_data)
result_label = ttk.Label(root, text="")

# Place labels, button, and result label on the main window
welcome_label.grid(row=0, column=0, columnspan=2, pady=(20, 10))
instruction_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
process_button.grid(row=2, column=0, padx=(50, 0), pady=(0, 20))
result_label.grid(row=2, column=1, padx=(20, 0), pady=(0, 20))

# Run the main loop
root.mainloop()