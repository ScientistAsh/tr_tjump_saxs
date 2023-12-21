# Define the file name prefixes for the time delays CH505
file_name_prefixes_Buffer_1 = ['buffer_20hs_44C_-10us_', 'buffer_20hs_44C_-5us_',
							'buffer_20hs_44C_10us_', 'buffer_20hs_44C_50us_', 'buffer_20hs_44C_100us_',
							'buffer_20hs_44C_500us_',
							'buffer_20hs_44C_1ms_']

file_name_prefixes_Buffer_2 = ['buffer_20hz_44C_-10us_', 'buffer_20hz_44C_-5us_',
							'buffer_20hz_44C_1.5us_', 'buffer_20hz_44C_3us_', 'buffer_20hz_44C_5us_']

file_name_prefixes_Buffer_3 = ['buffer_5hz_44C_-10us_', 'buffer_5hz_44C_-5us_',
							'buffer_5hz_44C_1ms_', 'buffer_5hz_44C_10ms_', 'buffer_5hz_44C_100ms_']
														
file_name_prefixes_Protein_1 = ['protein_20hz_set01_-10us_', 'protein_20hz_set01_-5us_',
							'protein_20hz_set01_10us_', 'protein_20hz_set01_50us_', 'protein_20hz_set01_100us_',
							'protein_20hz_set01_500us_',
							'protein_20hz_set01_1ms_']
				
file_name_prefixes_Protein_2 = ['protein_20hs_44C_-10us_', 'protein_20hs_44C_-5us_',
							'protein_20hs_44C_1.5us_', 'protein_20hs_44C_3us_', 'protein_20hs_44C_5us_']			

file_name_prefixes_Protein_3 = ['protein_5hz_set01_-10us_', 'protein_5hz_set01_-5us_',
							'protein_5hz_set01_1ms_', 'protein_5hz_set01_10ms_', 'protein_5hz_set01_100ms_']	
							
# Read the files and organize the data by time delay CH505
data_by_time_delay_Protein_1 = read_files('protein_20hz_set01/processedb', file_name_prefixes_Protein_1)
data_by_time_delay_Protein_2 = read_files('protein_20hz_set02/processed', file_name_prefixes_Protein_2)
data_by_time_delay_Protein_3 = read_files('protein_5hz_set01/processdb', file_name_prefixes_Protein_3)
data_by_time_delay_Buffer_1 = read_files('bs_5us-1ms_redo/processed', file_name_prefixes_Buffer_1)
data_by_time_delay_Buffer_2 = read_files('bs_1.5-5us_redo/processed', file_name_prefixes_Buffer_2)
data_by_time_delay_Buffer_3 = read_files('bs_1ms-100ms_redo/processed', file_name_prefixes_Buffer_3)

# Calculate the difference curves for each time delay
diff_curves_set_1_Protein_1, diff_curves_set_2_Protein_1 = calculate_difference_curves(data_by_time_delay_Protein_1)
diff_curves_set_1_Protein_2, diff_curves_set_2_Protein_2 = calculate_difference_curves(data_by_time_delay_Protein_2)
diff_curves_set_1_Protein_3, diff_curves_set_2_Protein_3 = calculate_difference_curves(data_by_time_delay_Protein_3)
diff_curves_set_1_Buffer_1, diff_curves_set_2_Buffer_1 = calculate_difference_curves(data_by_time_delay_Buffer_1)
diff_curves_set_1_Buffer_2, diff_curves_set_2_Buffer_2 = calculate_difference_curves(data_by_time_delay_Buffer_2)
diff_curves_set_1_Buffer_3, diff_curves_set_2_Buffer_3 = calculate_difference_curves(data_by_time_delay_Buffer_3)

# Perform outlier detection and removal on the entire diff_curves_set_1 and diff_curves_set_2
filtered_diff_curves_set_1_P1 = detect_and_remove_outliers(diff_curves_set_1_Protein_1)
filtered_diff_curves_set_1_P2 = detect_and_remove_outliers(diff_curves_set_1_Protein_2)
filtered_diff_curves_set_1_P3 = detect_and_remove_outliers(diff_curves_set_1_Protein_3)
filtered_diff_curves_set_1_B1 = detect_and_remove_outliers(diff_curves_set_1_Buffer_1)
filtered_diff_curves_set_1_B2 = detect_and_remove_outliers(diff_curves_set_1_Buffer_2)
filtered_diff_curves_set_1_B3 = detect_and_remove_outliers(diff_curves_set_1_Buffer_3)

# Perform iterative chi-square test on filtered_diff_curves_set_1 and filtered_diff_curves_set_2
filtered_diff_curves_set_1_P1 = iterative_chi_square_test(filtered_diff_curves_set_1_P1, chi_square_cutoff=1.5)
filtered_diff_curves_set_1_P2 = iterative_chi_square_test(filtered_diff_curves_set_1_P2, chi_square_cutoff=1.5)
filtered_diff_curves_set_1_P3 = iterative_chi_square_test(filtered_diff_curves_set_1_P3, chi_square_cutoff=1.5)
filtered_diff_curves_set_1_B1 = iterative_chi_square_test(filtered_diff_curves_set_1_B1, chi_square_cutoff=1.5)
filtered_diff_curves_set_1_B2 = iterative_chi_square_test(filtered_diff_curves_set_1_B2, chi_square_cutoff=1.5)
filtered_diff_curves_set_1_B3 = iterative_chi_square_test(filtered_diff_curves_set_1_B3, chi_square_cutoff=1.5)

# Get averaged curves for filtered_diff_curves_set_1 and filtered_diff_curves_set_2
averaged_curves_set_1_P1 = get_averaged_curves(filtered_diff_curves_set_1_P1)
averaged_curves_set_1_P2 = get_averaged_curves(filtered_diff_curves_set_1_P2)
averaged_curves_set_1_P3 = get_averaged_curves(filtered_diff_curves_set_1_P3)
averaged_curves_set_1_B1 = get_averaged_curves(filtered_diff_curves_set_1_B1)
averaged_curves_set_1_B2 = get_averaged_curves(filtered_diff_curves_set_1_B2)
averaged_curves_set_1_B3 = get_averaged_curves(filtered_diff_curves_set_1_B3)

# Scale the buffer to the protein and subtract
x_range_min = 1.5
x_range_max = 2.5
scaled_and_subtracted1 = scale_and_subtract_curves(averaged_curves_set_1_P1, averaged_curves_set_1_B1, x_range_min, x_range_max)
scaled_and_subtracted2 = scale_and_subtract_curves(averaged_curves_set_1_P2, averaged_curves_set_1_B2, x_range_min, x_range_max)
scaled_and_subtracted3 = scale_and_subtract_curves(averaged_curves_set_1_P3, averaged_curves_set_1_B3, x_range_min, x_range_max)

# Combine 1.5-5us and 10us-1ms
combined_set_P = filtered_diff_curves_set_1_P2 + filtered_diff_curves_set_1_P1
combined_set_B = averaged_curves_set_1_B2 + averaged_curves_set_1_B1
combined_scalesub_set_P = scaled_and_subtracted2 + scaled_and_subtracted1

# Plot the average curves with standard errors for averaged_curves_set_1
# plot_averaged_curves(combined_scalesub_set_P, 'All Curves')

# Bootstrap Analysis
# boots = bootstrap_difference_curves(combined_set_P, combined_set_B, n_iterations=1000)
# analysis_results = analyze_bootstrap_results(boots)
# print(analysis_results)

# SVD Analysis
a, b = 0.025, 1.0  # example values, adjust as needed

# Extract windowed data
windowed_data_matrix = extract_windowed_data(combined_scalesub_set_P, a, b)

# Usage
U, s, Vt = perform_SVD(windowed_data_matrix)
plot_U_curves(U, n=3)

np.savetxt("U_SVD.csv", U, delimiter=",")
np.savetxt("s_SVD.csv", s, delimiter=",")
np.savetxt("Vt_SVD.csv", Vt, delimiter=",")

save_to_csv(scaled_and_subtracted1, 'protein_20hz_set01_processed.csv')
save_to_csv(scaled_and_subtracted2, 'protein_20hz_set02_processed.csv')
save_to_csv(scaled_and_subtracted3, 'protein_5hz_set01_processed.csv')

















