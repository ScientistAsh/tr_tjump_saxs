# Define the file name prefixes for the time delays CH848
file_name_prefixes_Protein_1 = ['TrimerOnly_TSeries-14_44C_-10us_', 'TrimerOnly_TSeries-14_44C_-5us_', 'TrimerOnly_TSeries-14_44C_5us_',
							'TrimerOnly_TSeries-14_44C_10us_', 'TrimerOnly_TSeries-14_44C_50us_', 'TrimerOnly_TSeries-14_44C_100us_',
							'TrimerOnly_TSeries-14_44C_250us_', 'TrimerOnly_TSeries-14_44C_500us_', 'TrimerOnly_TSeries-14_44C_750us_',
							'TrimerOnly_TSeries-14_44C_1ms_']
							
file_name_prefixes_Protein_2 = ['TrimerOnly_TSeries-15_44C_-10us_', 'TrimerOnly_TSeries-15_44C_-5us_', 'TrimerOnly_TSeries-15_44C_1ms_',
							'TrimerOnly_TSeries-15_44C_10ms_', 'TrimerOnly_TSeries-15_44C_100ms_']

file_name_prefixes_Protein_3 = ['TrimerOnly_Series-3_44C_44C_-10us_', 'TrimerOnly_Series-3_44C_44C_-5us_', 'TrimerOnly_Series-3_44C_44C_5us_',
							'TrimerOnly_Series-3_44C_44C_10us_', 'TrimerOnly_Series-3_44C_44C_50us_', 'TrimerOnly_Series-3_44C_44C_100us_',
							'TrimerOnly_Series-3_44C_44C_500us_', 'TrimerOnly_Series-3_44C_44C_1ms_']

file_name_prefixes_Buffer_1 = ['Buffer_TSeries-1_44C_-10us_', 'Buffer_TSeries-1_44C_-5us_', 'Buffer_TSeries-1_44C_5us_',
							'Buffer_TSeries-1_44C_10us_', 'Buffer_TSeries-1_44C_50us_', 'Buffer_TSeries-1_44C_100us_',
							'Buffer_TSeries-1_44C_250us_', 'Buffer_TSeries-1_44C_500us_', 'Buffer_TSeries-1_44C_750us_',
							'Buffer_TSeries-1_44C_1ms_']
							
file_name_prefixes_Buffer_2 = ['Buffer_TSeries-2_44C_-10us_', 'Buffer_TSeries-2_44C_-5us_', 'Buffer_TSeries-2_44C_1ms_',
							'Buffer_TSeries-2_44C_10ms_', 'Buffer_TSeries-2_44C_100ms_']
							
# Read the files and organize the data by time delay CH505
data_by_time_delay_Protein_1 = read_files('TrimerOnly_TSeries-14_44C/processed', file_name_prefixes_Protein_1)
data_by_time_delay_Protein_2 = read_files('TrimerOnly_TSeries-15_44C/processed', file_name_prefixes_Protein_2)
data_by_time_delay_Protein_3 = read_files('Trimer10.17Only_Series3_44C/processed_series3_44C', file_name_prefixes_Protein_3)
data_by_time_delay_Buffer_1 = read_files('Buffer_TSeries-1_44C/processed', file_name_prefixes_Buffer_1)
data_by_time_delay_Buffer_2 = read_files('Buffer_TSeries-2_44C/processed', file_name_prefixes_Buffer_2)

# Calculate the difference curves for each time delay
diff_curves_set_1_Protein_1, diff_curves_set_2_Protein_1 = calculate_difference_curves(data_by_time_delay_Protein_1)
diff_curves_set_1_Protein_2, diff_curves_set_2_Protein_2 = calculate_difference_curves(data_by_time_delay_Protein_2)
diff_curves_set_1_Protein_3, diff_curves_set_2_Protein_3 = calculate_difference_curves(data_by_time_delay_Protein_3)
diff_curves_set_1_Buffer_1, diff_curves_set_2_Buffer_1 = calculate_difference_curves(data_by_time_delay_Buffer_1)
diff_curves_set_1_Buffer_2, diff_curves_set_2_Buffer_2 = calculate_difference_curves(data_by_time_delay_Buffer_2)

# Combine Series-3 and Series-14 curves
diff_curves_set_2_Protein_1_3 = merge_datasets(diff_curves_set_2_Protein_1, diff_curves_set_2_Protein_3)

# Perform outlier detection and removal on the entire diff_curves_set_1 and diff_curves_set_2
filtered_diff_curves_set_2_P1 = detect_and_remove_outliers(diff_curves_set_2_Protein_1)
filtered_diff_curves_set_2_P2 = detect_and_remove_outliers(diff_curves_set_2_Protein_2)
filtered_diff_curves_set_2_P1_P3 = detect_and_remove_outliers(diff_curves_set_2_Protein_1_3)
filtered_diff_curves_set_2_B1 = detect_and_remove_outliers(diff_curves_set_2_Buffer_1)
filtered_diff_curves_set_2_B2 = detect_and_remove_outliers(diff_curves_set_2_Buffer_2)

# Perform iterative chi-square test on filtered_diff_curves_set_1 and filtered_diff_curves_set_2
filtered_diff_curves_set_2_P1 = iterative_chi_square_test(filtered_diff_curves_set_2_P1, chi_square_cutoff=1.5)
filtered_diff_curves_set_2_P2 = iterative_chi_square_test(filtered_diff_curves_set_2_P2, chi_square_cutoff=1.5)
filtered_diff_curves_set_2_P1_P3 = iterative_chi_square_test(filtered_diff_curves_set_2_P1_P3, chi_square_cutoff=1.5)
filtered_diff_curves_set_2_B1 = iterative_chi_square_test(filtered_diff_curves_set_2_B1, chi_square_cutoff=1.5)
filtered_diff_curves_set_2_B2 = iterative_chi_square_test(filtered_diff_curves_set_2_B2, chi_square_cutoff=1.5)

# Get averaged curves for filtered_diff_curves_set_1 and filtered_diff_curves_set_2
averaged_curves_set_2_P1 = get_averaged_curves(filtered_diff_curves_set_2_P1)
averaged_curves_set_2_P2 = get_averaged_curves(filtered_diff_curves_set_2_P2)
averaged_curves_set_2_P1_P3 = get_averaged_curves(filtered_diff_curves_set_2_P1_P3)
averaged_curves_set_2_B1 = get_averaged_curves(filtered_diff_curves_set_2_B1)
averaged_curves_set_2_B2 = get_averaged_curves(filtered_diff_curves_set_2_B2)

smoothed_averaged_curves_B1 = apply_savgol_to_averaged_curves(averaged_curves_set_2_B1, window_length=250, polyorder=3)
smoothed_averaged_curves_B2 = apply_savgol_to_averaged_curves(averaged_curves_set_2_B2, window_length=250, polyorder=3)

# Scale the buffer to the protein and subtract
x_range_min = 0.5
x_range_max = 1.0
scaled_and_subtracted_1 = scale_and_subtract_curves(averaged_curves_set_2_P1, smoothed_averaged_curves_B1, x_range_min, x_range_max)
scaled_and_subtracted_2 = scale_and_subtract_curves(averaged_curves_set_2_P2, smoothed_averaged_curves_B2, x_range_min, x_range_max)
scaled_and_subtracted_1_3 = scale_and_subtract_curves(averaged_curves_set_2_P1_P3, smoothed_averaged_curves_B1, x_range_min, x_range_max)

# Scale the buffer to the protein and subtract
x_range_min =1.5
x_range_max = 2.5
scaled_and_subtracted_1_3_2 = scale_and_subtract_curves(averaged_curves_set_2_P1_P3, smoothed_averaged_curves_B1, x_range_min, x_range_max)


# Plot the average curves with standard errors for averaged_curves_set_1
plot_average_curves_with_error(scaled_and_subtracted_1_3, scaled_and_subtracted_1_3_2, x_range=[0.025, 1.0])

# Bootstrap Analysis
# boots = bootstrap_difference_curves(filtered_diff_curves_set_2_P1_P3, smoothed_averaged_curves_B1, n_iterations=1000, x_range_min=0.5, x_range_max=1.0)
# analysis_results = analyze_bootstrap_results(boots)
# print(analysis_results)

# SVD Analysis
a, b = 0.025, 1.0  # example values, adjust as needed

# Extract windowed data
windowed_data_matrix = extract_windowed_data(scaled_and_subtracted_1_3, a, b)

# Usage
U, s, Vt = perform_SVD(windowed_data_matrix)
plot_U_curves(U, n=3)

np.savetxt("U_SVD.csv", U, delimiter=",")
np.savetxt("s_SVD.csv", s, delimiter=",")
np.savetxt("Vt_SVD.csv", Vt, delimiter=",")

save_to_csv(scaled_and_subtracted_1, 'TrimerOnly_TSeries-14_44C_processed.csv')
save_to_csv(scaled_and_subtracted_2, 'TrimerOnly_TSeries-15_44C_processed.csv')
save_to_csv(scaled_and_subtracted_1_3, 'TrimerOnly_TSeries-14-3_44C_processed.csv')

