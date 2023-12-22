from scipy import stats
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import differential_evolution

def error_function_single(params):
    return np.sum((y_data - single_exp(x_data, *params))**2)

def error_function_double(params):
    return np.sum((y_data - double_exp(x_data, *params))**2)
    
def single_exp(x, A, tau, B):
    return A * np.exp(-x/tau) + B

def double_exp(x, A1, tau1, A2, tau2, B):
    return A1 * np.exp(-x/tau1) + A2 * np.exp(-x/tau2) + B

# SVD CH848
x_data = np.asarray([5, 10, 50, 100, 250, 500, 750, 1000])
y_data = Vt[0]

# AUC CH848
# x_data = np.asarray([5, 10, 50, 100, 250, 500, 750, 1000])
# y_data = analysis_results_1['mean_AUC']

# SVD CH505
# x_data = np.asarray([1.5, 3.0, 5, 10, 50, 100, 500, 1000])
# y_data = Vt[0]

# # AUC CH505
# x_data = np.asarray([1.5, 3.0, 5, 10, 50, 100, 500, 1000])
# y_data = analysis_results['mean_AUC']

# SVD CH505 w/o 1.5 and 3 us
# x_data = np.asarray([5, 10, 50, 100, 500, 1000])
# y_data = vt0

# # AUC CH505  w/o 1.5 and 3 us
# x_data = np.asarray(5, 10, 50, 100, 500, 1000])
# y_data = analysis_results['mean_AUC']

# Fit data

# Assuming bounds for each parameter in the form (min, max)
bounds_single = [(-1, 1), (0, 500), (0, 1)]
bounds_double = [(-1, 0), (0, 5), (-1, 0), (100, 300), (0, 1)]

result_single = differential_evolution(error_function_single, bounds_single)
result_double = differential_evolution(error_function_double, bounds_double)

popt_single, pcov_single = curve_fit(single_exp, x_data, y_data, p0=result_single.x)
popt_double, pcov_double = curve_fit(double_exp, x_data, y_data, p0=result_double.x) #p0=np.asarray([-0.3275954079774702 , 2.2159983454013608, -0.14719129535402503, 260.4105138274997, 0.4269729346910296]))

alpha = 0.05  # 95% confidence interval = 100*(1-alpha)

n = len(y_data)    # number of data points
p = len(popt_single)  # number of parameters

dof = max(0, n - p)  # number of degrees of freedom

# student-t value for the dof and confidence level
t_val = stats.t.ppf(1.0-alpha/2., dof) 

sigma_single = np.diag(pcov_single)**0.5
conf_int_single = t_val * sigma_single/np.sqrt(n)

sigma_double = np.diag(pcov_double)**0.5
conf_int_double = t_val * sigma_double/np.sqrt(n)

alpha = 0.05  # 95% confidence interval = 100*(1-alpha)

n = len(y_data)    # number of data points
p = len(popt_single)  # number of parameters

dof = max(0, n - p)  # number of degrees of freedom

# student-t value for the dof and confidence level
t_val = stats.t.ppf(1.0-alpha/2., dof) 

sigma_single = np.diag(pcov_single)**0.5
conf_int_single = t_val * sigma_single/np.sqrt(n)

sigma_double = np.diag(pcov_double)**0.5
conf_int_double = t_val * sigma_double/np.sqrt(n)

print("Single Exponential Fit:")
print("A =", popt_single[0], "±", conf_int_single[0])
print("tau =", popt_single[1], "±", conf_int_single[1])
print("B =", popt_single[2], "±", conf_int_single[2])

print("\nDouble Exponential Fit:")
print("A1 =", popt_double[0], "±", conf_int_double[0])
print("tau1 =", popt_double[1], "±", conf_int_double[1])
print("A2 =", popt_double[2], "±", conf_int_double[2])
print("tau2 =", popt_double[3], "±", conf_int_double[3])
print("B =", popt_double[4], "±", conf_int_double[4])

plt.figure(figsize=(10,6))
plt.scatter(x_data, y_data, color='red', label="Data")
plt.plot(x_data, single_exp(x_data, *popt_single), color='blue', label="Single Exponential Fit")
plt.plot(x_data, double_exp(x_data, *popt_double), color='green', label="Double Exponential Fit")
plt.legend()
plt.show()

# Save file command
with open('Trimer_Run2-SVD_Vt_0_single_exp_model.pkl', 'wb') as f:
	pickle.dump((single_exp, popt_single, pcov_single), f)

df = pd.DataFrame({'Parameters': popt_single})  # or popt_double
df.to_csv('Trimer_Run2-SVD_Vt_0_fitted_parameters.csv', index=False)

with open('Trimer_Run2-SVD_Vt_0_double-_exp_model.pkl', 'wb') as f:
	pickle.dump((double_exp, popt_double, pcov_double), f)

df = pd.DataFrame({'Parameters': popt_double})  # or popt_double
df.to_csv('Trimer_Run2-SVD_Vt_0_double-fitted_parameters.csv', index=False)

# with open('Trimer_Run2-AUC_0_single_exp_model.pkl', 'wb') as f:
# 	pickle.dump((single_exp, popt_single, pcov_single), f)
# 
# df = pd.DataFrame({'Parameters': popt_single})  # or popt_double
# df.to_csv('Trimer_Run2-AUC_fitted_parameters.csv', index=False)
# 
# with open('Trimer_Run2-AUC_double-_exp_model.pkl', 'wb') as f:
# 	pickle.dump((double_exp, popt_double, pcov_double), f)
# 
# df = pd.DataFrame({'Parameters': popt_double})  # or popt_double
# df.to_csv('Trimer_Run2-AUC_double-fitted_parameters.csv', index=False)












