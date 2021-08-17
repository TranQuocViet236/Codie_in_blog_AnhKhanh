# Nomal distribution by StandardScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta
from scipy.stats import shapiro
import statsmodels.api as sm
import numpy as np

# #Create a beta distribution sequence
# data = beta(1, 10).rvs(1000).reshape((-1.1))
# print('data shape:' +  data.shape)