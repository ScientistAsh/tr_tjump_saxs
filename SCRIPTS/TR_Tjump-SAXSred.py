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
from scipy.signal import savgol_filter

def read_files(directory_path, file_name_prefixes, suffix='_Q', extension='chi'):
    data_by_time_delay = []

    all_files = os.listdir(directory_path)

    for prefix in file_name_prefixes:
        data_for_prefix = {}
        
        # Create the pattern for the current prefix
        pattern = re.compile(re.escape(prefix) + r'(\d{3})' + re.escape(suffix) + r'\.' + re.escape(extension))
        
        # Iterate over files in the directory
        for file_name in all_files:
            match = pattern.match(file_name)
            
            if match:
                file_number = match.group(1)  # Extract the file number from the matched pattern
                
                file_path = os.path.join(directory_path, file_name)
                with open(file_path, 'r') as f:
                    file_data = []
                    for line in f:
                        if line.strip():
                            try:
                                col1, col2 = map(float, line.split())
                                if np.isfinite(col1) and np.isfinite(col2):
                                    file_data.append((col1, col2))
                            except ValueError:
                                continue
                    data_for_prefix[file_number] = file_data
        
        data_by_time_delay.append(data_for_prefix)
    
    return data_by_time_delay

def calculate_difference_curves(data_by_time_delay):
    if len(data_by_time_delay) < 2:
        raise ValueError("At least two time delay curves are required.")
    
    # Extract the reference curves
    reference_curve_1 = data_by_time_delay[0]
    reference_curve_2 = data_by_time_delay[1]
    
    # Initialize lists to store the difference curves
    diff_curves_set_1 = []
    diff_curves_set_2 = []
    
    # Loop through the remaining curves and calculate the difference curves
    for i in range(2, len(data_by_time_delay)):
        current_curve = data_by_time_delay[i]
        diff_curve_1 = {}
        diff_curve_2 = {}
        
        for file_number, curve_data in current_curve.items():
            ref_curve_data_1 = reference_curve_1.get(file_number, [])
            ref_curve_data_2 = reference_curve_2.get(file_number, [])
            
            # Calculate difference curve for set 1
            if ref_curve_data_1:
                diff_curve_1[file_number] = [(x, y - y_ref) for (x, y), (_, y_ref) in zip(curve_data, ref_curve_data_1)]
            
            # Calculate difference curve for set 2
            if ref_curve_data_2:
                diff_curve_2[file_number] = [(x, y - y_ref) for (x, y), (_, y_ref) in zip(curve_data, ref_curve_data_2)]
        
        diff_curves_set_1.append(diff_curve_1)
        diff_curves_set_2.append(diff_curve_2)
    
    return diff_curves_set_1, diff_curves_set_2

def merge_datasets(dataset1, dataset2):
	# This is specific to the CH848 dataset. Will need to adjust if working with a different dataset.
    merged_dataset = []

    # Loop through each time delay in Dataset 1
    for delay in range(1, 9):  # Loop from time delays 1 through 8
        curve_data_1 = dataset1[delay - 1]
        merged_data_for_delay = curve_data_1.copy()  # Copy the data for the current delay from dataset1
        
        # If this delay exists in dataset 2, merge the data
        if delay in [1, 2, 3, 4, 6, 8]:
            if delay <= 4:
                curve_data_2 = dataset2[delay - 1]
            elif delay == 6:
                curve_data_2 = dataset2[delay - 2]
            elif delay == 8:
                curve_data_2 = dataset2[delay - 3]

            # For each file_number, concatenate the data from dataset2 to dataset1
            for file_number, data in curve_data_2.items():
                if file_number in merged_data_for_delay:
                    merged_data_for_delay[file_number].extend(data)
                else:
                    merged_data_for_delay[file_number] = data

        merged_dataset.append(merged_data_for_delay)

    return merged_dataset

def detect_systematic_changes(diff_curves_set):
    results = []
    
    for curves_dict in diff_curves_set:
        # Extract the mean y-values for each curve in the dataset
        mean_y_values = [np.mean([y for _, y in curve_data]) for curve_data in curves_dict.values()]
        
        # Create an array of curve indices corresponding to the mean y-values
        curve_indices = np.arange(len(mean_y_values))
        
        # Perform linear regression on the mean y-values as a function of the curve index
        slope, intercept, r_value, _, _ = linregress(curve_indices, mean_y_values)
        
        # Calculate R-squared value
        r_squared = r_value ** 2
        
        # Perform Mann-Kendall test
        tau, p_value = kendalltau(curve_indices, mean_y_values)
        
        # Append the results for the current dataset to the results list
        results.append((slope, intercept, r_squared, tau, p_value))
    
    return results

def detect_and_remove_outliers(diff_curves_set):
    filtered_diff_curves_set = []
    for scattering_curves in diff_curves_set:
        # Filter out any empty or invalid curves
        valid_curve_data = [(fn, curve_data) for fn, curve_data in scattering_curves.items() if curve_data]
        
        # Extract file numbers and curve data from the valid curves
        file_numbers, curve_data_list = zip(*valid_curve_data)
        
        # Create a matrix where each row represents the y-values of a scattering curve
        matrix = np.vstack([np.array([y for _, y in curve_data]) for curve_data in curve_data_list])

        # Perform singular value decomposition (SVD)
        U, S, VT = np.linalg.svd(matrix, full_matrices=False)

        # Calculate the mean and standard deviation of the first column of U (u1,n)
        mean_u1n = np.mean(U[:, 0])
        std_u1n = np.std(U[:, 0])

        # Identify the rows where u1,n is within the acceptable range
        mask = np.abs(U[:, 0] - mean_u1n) <= 2.5 * std_u1n

        # Filter out the outliers
        filtered_matrix = matrix[mask]
        filtered_file_numbers = [fn for fn, is_valid in zip(file_numbers, mask) if is_valid]
        # Convert each row of filtered_matrix back into the original curve format
        filtered_scattering_curves = {
            file_number: list(zip(x_values, y_values))
            for file_number, (curve_data, y_values) in zip(filtered_file_numbers, zip(curve_data_list, filtered_matrix))
            if len(curve_data) == len(y_values)  # Ensure x and y have the same length
            for x_values, _ in [zip(*curve_data)]  # Extract x-values from the original curve_data
        }

        # Count the number of removed curves
        removed_count = len(scattering_curves) - len(filtered_scattering_curves)

        # Print the number of curves removed
        print(f"Removed {removed_count} outlier curves.")

        # Add the filtered scattering curves to the result set
        filtered_diff_curves_set.append(filtered_scattering_curves)

    return filtered_diff_curves_set

def iterative_chi_square_test(diff_curves_set, chi_square_cutoff=1.5):
    filtered_diff_curves_set = []
    
    for curve_index, curves_dict in enumerate(diff_curves_set):
        # Convert the dictionary to a list of curves
        curve_list = list(curves_dict.values())
        
        # Initialize the flag to indicate whether any outliers were removed in the current iteration
        outliers_removed = True
        
        # Initialize a variable to track the iteration number
        iteration = 0
        
        while outliers_removed:
            outliers_removed = False
            iteration += 1  # Increment the iteration number
            
            # Calculate the global average curve
            global_average_curve = np.nanmean(curve_list, axis=0)
            
            # Create a list to store the filtered curves
            filtered_curves = []
            
            # Initialize a variable to count the number of curves removed in the current iteration
            num_curves_removed = 0
            
            for curve_data in curve_list:
                # Calculate the chi-square value for each curve
                chi_square = np.nansum(((curve_data - global_average_curve) ** 2) / global_average_curve)
                
                # Check if the chi-square value exceeds the cutoff threshold
                if chi_square <= chi_square_cutoff:
                    filtered_curves.append(curve_data)
                else:
                    outliers_removed = True
                    num_curves_removed += 1
            
            # Print the current curve index, iteration number, and number of curves removed
            print(f"Curve Index: {curve_index+1}, Iteration: {iteration}, Number of Curves Removed: {num_curves_removed}")
            
            # Update the curve_list with the filtered curves
            curve_list = filtered_curves
        
        # Convert the filtered curve list back to a dictionary
        filtered_curves_dict = {file_number: curve_data for file_number, curve_data in zip(curves_dict.keys(), curve_list)}
        
        # Add the filtered curves dictionary to the result set
        filtered_diff_curves_set.append(filtered_curves_dict)
    
    return filtered_diff_curves_set

def get_averaged_curves(diff_curves_set):
    averaged_curves = []
    for index, curves_dict in enumerate(diff_curves_set):
        # Convert the dictionary to a list of curves
        curve_list = list(curves_dict.values())
        
        # Stack the curves into a matrix where each row represents the y-values of a curve
        matrix = np.vstack([np.array([y for _, y in curve_data]) for curve_data in curve_list])
        
        # Calculate the mean and standard error of y-values across all curves
        mean_y_values = np.nanmean(matrix, axis=0)
        std_errors = np.nanstd(matrix, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(matrix), axis=0))
        
        # Get the x-values from the first curve
        x_values = np.array([x for x, _ in curve_list[0]])
        
        # Store the averaged curve data in the result list
        averaged_curve_data = {
            'x_values': x_values,
            'mean_y_values': mean_y_values,
            'std_errors': std_errors
        }
        averaged_curves.append(averaged_curve_data)
    
    return averaged_curves

def compare_curves(averaged_curves_set_1, averaged_curves_set_2, significance_level=0.05):
    comparison_results = []
    for curve_data_1, curve_data_2 in zip(averaged_curves_set_1, averaged_curves_set_2):
        # Get the mean y-values from the averaged curves
        mean_y_values_1 = curve_data_1['mean_y_values']
        mean_y_values_2 = curve_data_2['mean_y_values']
        
        # Perform the Mann-Whitney U test
        u_statistic, p_value = mannwhitneyu(mean_y_values_1, mean_y_values_2)
        
        # Determine whether the two curves are significantly different
        are_different = p_value < significance_level
        
        # Store the comparison result in the result list
        comparison_result = {
            'u_statistic': u_statistic,
            'p_value': p_value,
            'are_different': are_different
        }
        comparison_results.append(comparison_result)
    
    return comparison_results

def rms_scale_within_range(reference, to_scale, x_values, x_min, x_max):
    """Scale the 'to_scale' array to 'reference' using RMS scaling within a specified x range."""
    mask = (x_values >= x_min) & (x_values <= x_max)
    scale_factor = np.sqrt(np.nanmean(reference[mask]**2) / np.nanmean(to_scale[mask]**2))
    return to_scale * scale_factor

def scale_and_subtract_curves(averaged_curves_1, averaged_curves_2, x_min=None, x_max=None):
    scaled_and_subtracted_curves = []

    # Ensure that there are equal numbers of curves in both datasets
    if len(averaged_curves_1) != len(averaged_curves_2):
        raise ValueError("The number of averaged curves in both datasets must be equal.")
    
    for curve_1, curve_2 in zip(averaged_curves_1, averaged_curves_2):
        # Extract data from the curve dictionaries
        x_values_1 = curve_1['x_values']
        y_values_1 = curve_1['mean_y_values']
        y_values_2 = curve_2['mean_y_values']
        std_errors_1 = curve_1['std_errors']
        std_errors_2 = curve_2['std_errors']

        # If an x range is provided, scale y-values based on that range, otherwise use full range
        if x_min is not None and x_max is not None:
            scaled_y_values_2 = rms_scale_within_range(y_values_1, y_values_2, x_values_1, x_min, x_max)
        else:
            scale_factor = np.sqrt(np.mean(y_values_1**2)) / np.sqrt(np.mean(y_values_2**2))
            scaled_y_values_2 = y_values_2 * scale_factor
			print(scale_factor)
        scaled_std_errors_2 = std_errors_2 * (np.nanmean(scaled_y_values_2) / np.nanmean(y_values_2))

        # Subtract the scaled second curve from the first curve
        subtracted_y_values = y_values_1 - scaled_y_values_2
        subtracted_std_errors = np.sqrt(std_errors_1**2 + scaled_std_errors_2**2)

        # Store the subtracted curve data in the result list
        subtracted_curve_data = {
            'x_values': x_values_1,
            'mean_y_values': subtracted_y_values,
            'std_errors': subtracted_std_errors
        }

        scaled_and_subtracted_curves.append(subtracted_curve_data)
    
    return scaled_and_subtracted_curves

def get_averaged_curves_portion(diff_curves_set):
    averaged_curves = []
    
    for index, curves_dict in enumerate(diff_curves_set):
        # Convert the dictionary to a list of curves
        curve_list = list(curves_dict.values())
        
        # Sample 80% of the curves
        num_to_sample = int(0.8 * len(curve_list))
        sampled_curves = random.sample(curve_list, num_to_sample)
        
        # Stack the curves into a matrix where each row represents the y-values of a sampled curve
        matrix = np.vstack([np.array([y for _, y in curve_data]) for curve_data in sampled_curves])
        
        # Calculate the mean and standard error of y-values across the sampled curves
        mean_y_values = np.nanmean(matrix, axis=0)
        std_errors = np.nanstd(matrix, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(matrix), axis=0))
        
        # Get the x-values from the first curve
        x_values = np.array([x for x, _ in curve_list[0]])
        
        # Store the averaged curve data in the result list
        averaged_curve_data = {
            'x_values': x_values,
            'mean_y_values': mean_y_values,
            'std_errors': std_errors
        }
        averaged_curves.append(averaged_curve_data)
    
    return averaged_curves

def apply_savgol_to_averaged_curves(averaged_curves, window_length=51, polyorder=3):
    smoothed_averaged_curves = []
    
    for averaged_curve_data in averaged_curves:
        x_values = averaged_curve_data['x_values']
        mean_y_values = averaged_curve_data['mean_y_values']
        std_errors = averaged_curve_data['std_errors']
        
        # Apply Savitzky-Golay filter to the mean y-values
        mean_y_values_smoothed = savgol_filter(mean_y_values, window_length, polyorder)
        
        smoothed_averaged_curve_data = {
            'x_values': x_values,
            'mean_y_values': mean_y_values_smoothed,
            'std_errors': std_errors
        }
        
        smoothed_averaged_curves.append(smoothed_averaged_curve_data)
    
    return smoothed_averaged_curves
