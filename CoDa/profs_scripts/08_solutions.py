import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycodamath as coda
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

print('\nExercise 8.1')
'''
We load the data, build the basis (using pycoda), ILR transform, and do a
linear regression. From the plot it should be obvious why we chose to fit
only from 1985 and forward.
'''

data = pd.read_csv('../Data/world_energy.csv', sep=';', index_col='year')

balances = [[1, 1, 1, -1], [1, 1, -1, 0], [1, -1, 0, 0]]
psi = coda.extra.norm(balances)
ilr = data.coda.ilr(psi)


reg = [LinearRegression().fit(np.array(ilr.index[5:]).reshape(-1, 1), ilr.iloc[5:][i])
       for i in [0, 1, 2]]

x = np.linspace(1960, 2040, 100)
plt.clf()
plt.plot(ilr.iloc[:, 0], color='purple', alpha=0.6, lw=4, label='Non-renewables / renewables')
plt.plot(ilr.iloc[:, 1], color='seagreen', alpha=0.6, lw=4, label='Fossil / Nuclear')
plt.plot(ilr.iloc[:, 2], color='teal', alpha=0.6, lw=4, label='Coal / Oil')
plt.plot(x, reg[0].coef_*x+reg[0].intercept_, lw=1, color='black')
plt.plot(x, reg[1].coef_*x+reg[1].intercept_, lw=1, color='black')
plt.plot(x, reg[2].coef_*x+reg[2].intercept_, lw=1, color='black')
plt.legend()
plt.xlabel('Year')
plt.ylabel('ILR coordinates')
plt.show()


print('\nExercise 8.2')

data = pd.read_csv('../Data/protein.csv', sep=';', index_col='Country')
response = data[data.columns[:-2]]
covariates = data[data.columns[-2:]]

# Sample mean of full data set
response.coda.gmean()

# Mean of the western samples
west = covariates[covariates['label'].str.contains('West')].index
west_center = response.loc[west].coda.gmean()

# Mean of the eastern samples
east = covariates[covariates['label'].str.contains('East')].index
east_center = response.loc[east].coda.gmean()


# perturbative difference
pert_diff = [west_center[i]/east_center[i] for i in range(len(west_center))]
# and close to 1
pert_diff = pert_diff/sum(pert_diff)


# beta2 from the notes
beta2 = [0.05, -0.26, 0.21, 0.06, 1.21, -0.63, 0.01, -0.54, -0.09]
# Inverse CLR-transform
x = np.exp(beta2)/sum(np.exp(beta2))

plt.clf()
plt.plot(x, pert_diff, 'x')

print('\nExercise 8.3')

data = pd.read_csv('../Data/protein.csv', sep=';', index_col='Country')
response = data[data.columns[:-2]]
covariates = data[data.columns[-2:]]


# Sorted CLRbeta1
clrbeta1 = [-0.61, -0.48, -0.46, -0.45, -0.05, -0.04, 0.47, 0.48, 1.14]

# Build informed basis

balances = [[1, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, -1, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, -1],
            [0, 1, 1, -1, 0, 0, -1, 0, 0],
            [-1, 1, 1, 1, -1, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, -1, 1, 0, -1],
            [1, 1, 1, 1, 1, 1, 1, -1, 1]]

psi = coda.extra.norm(balances)
ilr = response.coda.ilr(psi)

# Rename columns for ols to work
for i in ilr.columns:
    ilr = ilr.rename(columns={i: 'ilr'+str(i)})

# add covariates
ilr['covariates'] = covariates['label'].str[:5]

# Do ANOVA per part
for part in ilr.columns[:-1]:
    model = ols(part+' ~ covariates', data=ilr).fit()
    print(part)
    print('beta', model.params[1].round(2))
    print('t value', np.sqrt(model.fvalue).round(2))
    print('p value', model.pvalues[1].round(4))
    print()

'''
Nuts versus everything else and cereal-vegetables versus everything but nuts
are significantly over-consumed in the south.
'''

print('\nExercise 8.4')

data = pd.read_csv('../Data/protein.csv', sep=';', index_col='Country')
response = data[['Fish', 'Milk', 'Eggs', 'Red meat', 'White meat']]
covariates = data[data.columns[-2:]]

balances = [[1, 1, 1, 1, -1], [1, 1, 1, -1, 0], [1, 1, -1, 0, 0], [1, -1, 0, 0, 0]]
psi = coda.extra.norm(balances)
ilr = response.coda.ilr(psi)
ilr = ilr.rename(columns={0: 'ilr0', 1: 'ilr1', 2: 'ilr2', 3: 'ilr3'})
ilr['covariates'] = covariates['label'].str[:5]


beta = [ols(part + '~ covariates', data=ilr).fit().params[1] for part in ilr.columns[:-1]]
beta = pd.DataFrame(beta, columns=['beta'], index=ilr.columns[:-1]).T
clrbeta = pd.DataFrame(np.dot(beta, psi), index=['clrbeta'], columns=response.columns)
print(clrbeta.iloc[0].sort_values())
# group red meat on its own, fish and milk, eggs and white meat

balances = [[1, 1, 1, -1, 1], [-1, -1, 1, 0, 1], [-1, 1, 0, 0, 0], [0, 0, -1, 0, 1]]
psi = coda.extra.norm(balances)
ilr = response.coda.ilr(psi)
ilr = ilr.rename(columns={0: 'ilr0', 1: 'ilr1', 2: 'ilr2', 3: 'ilr3'})
ilr['covariates'] = covariates['label'].str[:5]

# Do ANOVA per part
for part in ilr.columns[:-1]:
    model = ols(part+' ~ covariates', data=ilr).fit()
    print(part)
    print('beta', model.params[1].round(2))
    print('t value', np.sqrt(model.fvalue).round(2))
    print('p value', model.pvalues[1].round(4))
    print()

# Now relabel samples and run again for a significant split.
# Try labeling the "nordic" countries as north and everything else as south.
