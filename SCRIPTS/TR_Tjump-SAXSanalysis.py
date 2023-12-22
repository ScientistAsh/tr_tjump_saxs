import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, linregress
from scipy.stats import mannwhitneyu
from scipy.optimize import curve_fit
import pandas as pd
import random
from scipy.integrate import simps

def extract_windowed_data(data_list, a, b):
    windowed_data_matrix = []

    for data_dict in data_list:
        x_values = data_dict['x_values']
        y_values = data_dict['mean_y_values']
        windowed_y_values = [y for x, y in zip(x_values, y_values) if a <= x <= b]
        windowed_data_matrix.append(windowed_y_values)

    return np.array(windowed_data_matrix).T

def perform_SVD(data_matrix):
    # Perform SVD on the provided matrix data
    U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    return U, s, Vt

# Define the single and double exponential functions
def single_exponential(t, A, k, C):
    return A * np.exp(-k * t) + C

def double_exponential(t, A1, k1, A2, k2, C):
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C
    
def save_to_csv(result_list, filename):
    # Convert the list of dictionaries into a pandas DataFrame
    dfs = []
    for i, curve_data in enumerate(result_list):
        df = pd.DataFrame({
            'x_values': curve_data['x_values'],
            f'mean_y_values_curve_{i}': curve_data['mean_y_values'],
            f'std_errors_curve_{i}': curve_data['std_errors']
        })
        dfs.append(df)

    # Merge DataFrames on 'x_values'
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='x_values', how='outer')

    # Save the DataFrame to a CSV file
    merged_df.to_csv(filename, index=False)

def calculate_AUC(scaled_and_subtracted_curves, x_min=None, x_max=None):
    """
    Calculate AUC for each scaled and subtracted curve using Simpson's rule within a specified x range.
    
    Parameters:
    - scaled_and_subtracted_curves: List of dictionaries containing x_values and mean_y_values for each curve.
    - x_min: Minimum x-value for the range.
    - x_max: Maximum x-value for the range.
    
    Returns:
    - A list containing AUC for each curve.
    """
    auc_values = []

    for curve in scaled_and_subtracted_curves:
        x_values = curve['x_values']
        y_values = np.abs(curve['mean_y_values'])  # Convert negative values to absolute
        
        # Filter x and y values based on the specified range
        if x_min is not None and x_max is not None:
            mask = (x_values >= x_min) & (x_values <= x_max)
            x_values = np.array(x_values)[mask]
            y_values = y_values[mask]

        auc = simps(y_values, x_values)  # Use Simpson's rule for integration
        auc_values.append(auc)

    return auc_values
    
def bootstrap_difference_curves(diff_curves_set_1, diff_curves_B, n_iterations=1000, x_range_min=None, x_range_max=None):

    bootstrap_results = []
    i=1
	
    for _ in range(n_iterations):
        print(i)
        i += 1
        resampled_set_1 = {}

        # Resample curves from set 1
        averaged_curves_1 = get_averaged_curves_portion(diff_curves_set_1)

        # Scale and subtract the curves
        final_curves = scale_and_subtract_curves(averaged_curves_1, diff_curves_B, x_range_min, x_range_max)

        # Extract Vt[0] and AUC from the final curves
        # Assuming you have a function get_Vt_0 and calculate_AUC which can extract the required data from the final curves

        # SVD Analysis
        a, b = 0.025, 1.0  # example values, adjust as needed

        # Extract windowed data
        # windowed_data_matrix = extract_windowed_data(averaged_curves_set_1_B, a, b)
        windowed_data_matrix = extract_windowed_data(final_curves, a, b)

        # Usage
        U, s, Vt = perform_SVD(windowed_data_matrix)
        AUC_data = calculate_AUC(final_curves, x_min=0.025, x_max=0.09)

        bootstrap_results.append({
            'Vt_0': Vt[0],
            'Vt_1': Vt[1],
            'AUC': AUC_data
        })

    return bootstrap_results

def get_mean_stderr(data):
    """Calculate mean and stderr for a list of data."""
    data_array = np.array(data)
    mean_values = np.nanmean(data_array, axis=0)
    stderr_values = np.nanstd(data_array, axis=0) / np.sqrt(data_array.shape[0])
    return mean_values, stderr_values

def analyze_bootstrap_results(bootstrap_results):
    # Extract Vt[0] and AUC values from the bootstrap results
    Vt_0_values = [result['Vt_0'] for result in bootstrap_results]
    Vt_1_values = [result['Vt_1'] for result in bootstrap_results]
    AUC_values = [result['AUC'] for result in bootstrap_results]

    # Calculate mean and stderr for Vt[0] values
    mean_Vt_0, stderr_Vt_0 = get_mean_stderr(Vt_0_values)

    # Calculate mean and stderr for Vt[0] values
    mean_Vt_1, stderr_Vt_1 = get_mean_stderr(Vt_1_values)
    
    # Calculate mean and stderr for AUC values
    mean_AUC, stderr_AUC = get_mean_stderr(AUC_values)

    return {
        'mean_Vt_0': mean_Vt_0,
        'stderr_Vt_0': stderr_Vt_0,
        'mean_Vt_1': mean_Vt_1,
        'stderr_Vt_1': stderr_Vt_1,
        'mean_AUC': mean_AUC,
        'stderr_AUC': stderr_AUC
    }
