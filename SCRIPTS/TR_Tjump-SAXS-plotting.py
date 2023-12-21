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

def plot_difference_curves(diff_curves_set_1, diff_curves_set_2, index):
    # Get the difference curves for the specified index
    diff_curves_1 = diff_curves_set_1[index]
    diff_curves_2 = diff_curves_set_2[index]
    
    num_curves = max(len(diff_curves_1), len(diff_curves_2))
    colormap = plt.cm.get_cmap('viridis', num_curves)
    
    # Create a figure with two subplots (one for each set of difference curves)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Difference Curves for Index {index+1}")
    
    # Plot the difference curves for set 1
    for idx, (file_number, curve_data) in enumerate(diff_curves_1.items()):
        x_values, y_values = zip(*curve_data)
        ax1.plot(x_values, y_values, color=colormap(idx), label=f"File {file_number}")
    ax1.set_title("Set 1")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend(loc="upper right")
    
    # Plot the difference curves for set 2
    for idx, (file_number, curve_data) in enumerate(diff_curves_2.items()):
        x_values, y_values = zip(*curve_data)
        ax2.plot(x_values, y_values, color=colormap(idx), label=f"File {file_number}")
    ax2.set_title("Set 2")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.legend(loc="upper right")
    
    # Show the plots
    plt.show()
    
def plot_mean_values_by_index(diff_curves_set, title=None):
    # Determine the number of subplots (one for each dataset)
    num_subplots = len(diff_curves_set)
    
    # Create a figure with the specified number of subplots arranged in a grid
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(6, 4 * num_subplots))
    
    # If there is only one subplot, convert axes to a list for consistency
    if num_subplots == 1:
        axes = [axes]
    
    # Loop through each dataset in the difference curves set and plot the mean values by index
    for ax, curves_dict in zip(axes, diff_curves_set):
        # Extract the mean y-values for each curve in the dataset
        mean_y_values = [np.mean([y for _, y in curve_data]) for curve_data in curves_dict.values()]
        
        # Create an array of curve indices corresponding to the mean y-values
        curve_indices = np.arange(len(mean_y_values))
        
        # Perform linear regression on the mean y-values as a function of the curve index
        slope, intercept, r_value, _, _ = linregress(curve_indices, mean_y_values)
        
        # Calculate the regression line
        regression_line = slope * curve_indices + intercept
        
        # Create a scatter plot of the mean y-values by index
        ax.scatter(curve_indices, mean_y_values, label='Mean y-values', alpha=0.7)
        
        # Plot the regression line
        ax.plot(curve_indices, regression_line, color='red', label='Regression line', linestyle='--')
        
        # Add labels and legend
        ax.set_xlabel('Curve Index')
        ax.set_ylabel('Mean Y-value')
        ax.legend()
    
    # Add a title to the figure
    if title:
        fig.suptitle(title)
    
    # Adjust the layout to prevent overlapping labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Show the plot
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_average_curves_with_error(averaged_curves_set_1, averaged_curves_set_2, x_range=None):
    # Create a figure with two subplots (one for each set of average curves)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Averaged Curves with Standard Errors")
    
    # Define a function to plot the average curve and standard error on the given axis
    def plot_curve_with_error(ax, averaged_curve, title, x_range=None):
        x_values = np.array(averaged_curve['x_values'])
        mean_y_values = np.array(averaged_curve['mean_y_values'])
        std_errors = np.array(averaged_curve['std_errors'])
        
        # Filter based on the provided x_range
        if x_range:
            min_x, max_x = x_range
            filtered_indices = (x_values >= min_x) & (x_values <= max_x)
            x_values = x_values[filtered_indices]
            mean_y_values = mean_y_values[filtered_indices]
            std_errors = std_errors[filtered_indices]

        ax.plot(x_values, mean_y_values, label='Mean Curve')
        ax.fill_between(x_values, mean_y_values - std_errors, mean_y_values + std_errors, alpha=0.3, label='Standard Error')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right")
    
    # Plot the average curves and standard errors for set 1
    for index, averaged_curve in enumerate(averaged_curves_set_1):
        plot_curve_with_error(ax1, averaged_curve, title=f"Set 1 - Curve {index+1}", x_range=x_range)
    
    # Plot the average curves and standard errors for set 2
    for index, averaged_curve in enumerate(averaged_curves_set_2):
        plot_curve_with_error(ax2, averaged_curve, title=f"Set 2 - Curve {index+1}", x_range=x_range)
    
    # Show the plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
    plt.show()

def plot_averaged_curves(averaged_curves, title=''):
    # Create a figure and set the number of rows to be equal to the number of curves
    fig, axs = plt.subplots(len(averaged_curves), 1, figsize=(10, 5 * len(averaged_curves)))
    
    # If there's only one subplot, axs won't be a list, so we make it a list for consistency
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    
    for ax, curve_data in zip(axs, averaged_curves):
        # Plot the mean y-values with error bars
        ax.errorbar(curve_data['x_values'], curve_data['mean_y_values'], yerr=curve_data['std_errors'], fmt='-')

        # Add labels to the x and y axes
        ax.set_xlabel('X Values')
        ax.set_ylabel('Mean Y Values')

        # Add a grid
        ax.grid(True)
    
    # Add a title to the figure
    fig.suptitle(title)

    # Adjust the layout so the plots do not overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Show the plot
    plt.show()
    
def plot_U_curves(U, n):
    # Plot the first n U curves
    plt.figure(figsize=(10, 6))
    
    for i in range(n):
        plt.plot(U[:, i], label=f"U-{i+1}")
        
    plt.title(f"First {n} U-curves from SVD")
    plt.xlabel("Windowed Data Point Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_V_points_against_time(Vt, times, n=2):
    if len(times) != Vt.shape[1]:
        raise ValueError("The length of times list must match the number of columns in Vt.")
    if n < 1:
        raise ValueError("Number of components to plot should be at least 1")

    plt.figure(figsize=(12, 6))
    
    # Plotting the points for the first component
    plt.subplot(1, 2, 1)
    plt.scatter(times, Vt[0, :], color='b', label="Component 1")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.title("Scores for Component 1")
    plt.legend()
    plt.grid(True)
    
    # Plotting the points for the second component
    if n > 1:
        plt.subplot(1, 2, 2)
        plt.scatter(times, Vt[1, :], color='r', label="Component 2")
        plt.xlabel("Time")
        plt.ylabel("Score")
        plt.title("Scores for Component 2")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
