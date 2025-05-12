import pandas as pd
import pycodamath as coda
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

plt.clf()
data = pd.read_csv('06_exercise_data.csv', sep=';', index_col=0)


print('\nExercise 6.1')

# Geometric center
gm = ss.mstats.gmean(data)
gm = 100/np.sum(gm) * gm
print(gm)
# [36.89924841850101, 36.44274286865587, 26.658008712843113]

# Variation matrix
npdata = np.array(data)
var_matrix = np.var(np.log(npdata[:, :, None] * 1./npdata[:, None]), axis=0)
print(var_matrix)
#   0.      1.904       1.209
#   1.904   0.          1.381
#   1.209   1.381       0

# Total variation
totvar = 1./(2 * 3) * np.sum(var_matrix)
print(totvar)
# 1.498


print('\nExercise 6.2')

# Perturb by inverse Gm
pert_data = data/gm
print(pert_data)
# Geometric mean: [ 33.33, 33.33, 33.33]
# var_matrix : same as above
# totvar : same as above

scaled_data = pow(data, 1./np.sqrt(totvar))
print(scaled_data)

print('\nExercise 6.3')
clr = pow(data/gm, 1./np.sqrt(totvar)).coda.clr()
s, e, l = np.linalg.svd(clr)
# scale loadings with eigenvalues
l = np.inner(e*np.identity(3), l.T[0:3, 0:3])

# plot
[plt.plot([0, l[0][i]], [0,l[1][i]], color='red') for i in [0, 1, 2]]
[plt.plot([s.T[0][i]], [s.T[1][i]], 'o', color='black') for i in range(20)]
