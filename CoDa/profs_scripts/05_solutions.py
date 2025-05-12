import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycodamath as coda
from pycoda import plot
from pycoda import extra
import ternary


data = pd.DataFrame(data={'x1': [79.07, 31.74, 18.61, 49.51, 29.22], 'x2': [
                    12.83, 56.69, 72.05, 15.11, 52.36], 'x3': [8.1, 11.57, 9.34, 35.38, 18.42]})


print('\nExercise 5.1')
data.plot.bar()
data.plot.bar(stacked=True)


print('\nExercise 5.2')
plt.figure(3)
plt.scatter(np.log(data['x1']/data['x2']), np.log(data['x2']/data['x3']))
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.plot([-2, 2], [2, -2], color='black')


print('\nExercise 5.3')
plot.ternary(data)


print('\nExercise 5.4')
psi = np.array([[1, -1, -1], [0, 1, -1]])
normpsi = extra.norm(psi)

ilr = data.coda.ilr(psi=normpsi)

plt.figure(5)
plt.plot(ilr[0], ilr[1], 'o')
plt.plot([-2.5, 2.5], [0, 0], '--', color='black')
plt.plot([0, 0], [-2.5, 2.5], '--', color='black')


print('\nExercise 5.5')
s = [0.1, 0.1, 0.8]

data_pert = data.copy()
for row in data.index:
    data_pert.loc[row] = [data.loc[row][i]*s[i] for i in range(3)]

ilr_pert = data_pert.coda.ilr(psi=normpsi)
plt.figure(6)
plt.plot(ilr[0], ilr[1], 'o')
plt.plot(ilr_pert[0], ilr_pert[1], 'o')
plt.plot([-2.5, 2.5], [0, 0], '--', color='black')
plt.plot([0, 0], [-2.5, 2.5], '--', color='black')

for i in data.index:
    plt.plot([ilr.loc[i][0], ilr_pert.loc[i][0]], [
             ilr.loc[i][1], ilr_pert.loc[i][1]], '-', color='black')


_, tax = ternary.figure(scale=100)
tax.boundary(linewidth=1.5)
tax.ticks(axis='lbr', linewidth=1, multiple=10, offset=0.03)
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')

tax.scatter(data.coda.closure(100).to_numpy(), color='steelblue', alpha=0.5)
tax.scatter(data_pert.coda.closure(100).to_numpy(), color='seagreen', alpha=0.5)

for i in range(5):
    x1 = np.linspace(ilr.loc[i][0], ilr_pert.loc[i][0], 100)
    x2 = np.linspace(ilr.loc[i][1], ilr_pert.loc[i][1], 100)
    ilr_line = list(zip(x1, x2))
    x = np.exp(np.matmul(ilr_line, normpsi))
    x = [100.*np.array([x[i][0], x[i][1], x[i][2]])/np.sum(x[i]) for i in range(len(x))]
    tax.plot(x, color='black', lw=0.5, ls=':')

ternary.plt.show()


print('\nExercise 5.6')
s = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
t = [0.7, 0.5, 0.8]

data_pow = pd.DataFrame(columns=['x1', 'x2', 'x3'])
for i in range(len(s)):
    data_pow.loc[i] = [pow(t[j], s[i]) for j in range(3)]


ilr_pow = data_pow.coda.ilr(psi=normpsi)
plt.figure(8)
plt.plot(ilr_pow[0], ilr_pow[1], 'o')
plt.plot([-2.5, 2.5], [0, 0], '--', color='black')
plt.plot([0, 0], [-2.5, 2.5], '--', color='black')

plot.ternary(data_pow.coda.closure(100))


plt.show()
