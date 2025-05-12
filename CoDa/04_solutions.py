import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sp
from pycodamath import extra
import scipy.optimize as optimization

plt.clf()


# Some functions to help you along
def closure(x, kappa):
    ''' Apply closure to x '''
    return kappa/sum(x)*x


def clr(x):
    ''' CLR transform x '''
    gmean = np.exp(1./len(x)*sum([np.log(i) for i in x]))
    return [np.log(i/gmean) for i in x]


def aitchison_mean(x, alpha):
    ''' Return the Aitchison mean point estimate '''
    return np.exp(sp.digamma(alpha+x))


print('\nExercise 4.1')

x = [0.18, 0.16, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.04, 0.02]

clr_values = clr(x)
print(np.round(clr_values, 2))

plt.plot(clr_values, 'o', color='orange', markersize=12)


print('\nExercise 4.2')

data = pd.read_csv('04_exercise_data.csv', index_col=0)
xaxis = np.linspace(0, 9, 10)

# Replacement by adding a pseudo-count. We choose a pseudo-count of 0.5
for sample in data.index:
    pseudocount = data.loc[sample].replace(0., 0.5)
    plt.plot(xaxis, clr(pseudocount), '.', color='black', alpha=0.3)


# Replacement using eq. 4.1 using a delta = 0.5
for sample in data.index:
    eq4d1 = [0.5 if i == 0. else i *
             (1-1./20 * 0.5*np.count_nonzero(data.loc[sample] == 0)) 
             for i in data.loc[sample]]
    plt.plot(xaxis+0.1, clr(eq4d1), '.', color='steelblue', alpha=0.3)

# Bayesian replacement using alpha = 0.5
for sample in data.index:
    bayesian = aitchison_mean(data.loc[sample], 0.5)
    plt.plot(xaxis+0.2, clr(bayesian), '.', color='seagreen', alpha=0.3)

# Bayesian replacement using alpha = 1.0
for sample in data.index:
    bayesian = aitchison_mean(data.loc[sample], 1.0)
    plt.plot(xaxis+0.3, clr(bayesian), '.', color='maroon', alpha=0.3)

# Iterative replacement

# Replace zeros in the first part. There are two zero values in the first part
# in sample 28 and 74

rplc = [28, 74]

# Temporary replace with fiducial value (1) so that we can ILR transform
tmp = data.replace(0., 1.)

# Build and normalize a basis, where we split part 1 in the first row
psi = [[1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
       [0,  1, -1, -1, -1, -1, -1, -1, -1, -1],
       [0,  0,  1, -1, -1, -1, -1, -1, -1, -1],
       [0,  0,  0,  1, -1, -1, -1, -1, -1, -1],
       [0,  0,  0,  0,  1, -1, -1, -1, -1, -1],
       [0,  0,  0,  0,  0,  1, -1, -1, -1, -1],
       [0,  0,  0,  0,  0,  0,  1, -1, -1, -1],
       [0,  0,  0,  0,  0,  0,  0,  1, -1, -1],
       [0,  0,  0,  0,  0,  0,  0,  0,  1, -1]]
# I normalize using the pycoda package to save some time
normpsi = extra.norm(psi)

# ILR transform (again using pycoda)
ilr = tmp.coda.ilr(psi=normpsi)

# Exclude the two samples with zero values in the first part
ilr_tmp = ilr.drop(rplc, axis=0)

# Do a least squares fit using the first part as y-values and the other parts as x-values


def func(params, xdata, ydata):
    ''' Dummy function for least squares fit '''
    return (ydata - np.dot(xdata, params))


x0 = np.array([1., 1., 1., 1., 1., 1., 1., 1])
ydata = np.array(ilr_tmp[0])
xdata = np.transpose(np.array(ilr_tmp.drop(0, axis=1).T))
par = optimization.leastsq(func, x0, args=(xdata, ydata))

# Replace values using fitted paramters
for i in rplc:
    ilr.iloc[i] = np.dot(par[0], ilr.drop(0, axis=1).iloc[i])

# Back-transform to the simplex
data_updated = ilr.coda.ilr_inv(psi=normpsi).coda.closure(1.)
