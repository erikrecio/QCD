#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import scienceplots
import latex
import rsmf

# colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']

fontfig = 9
column_width = 246 * 0.01389

mksize = 4.5
lwfig = 1.

plt.rcParams.update({'font.size': fontfig})
plt.rcParams.update({'font.family': 'times'})


with plt.style.context(['science', 'std-colors']):
    
    plt.rcParams['axes.linewidth'] = 1.10
    # fig, axis = plt.subplots(figsize =(4.8 / 6.4 * column_width, column_width))
    fig, axis = plt.subplots(1,1)
    fig.set_figheight(5)
    fig.set_figwidth(6)
    fig.tight_layout()
    
    x = 1
    delta = 1

    n = 100
    l = 1
    p = 0.1
    L0 = int(1+np.log(2*n)/(2*(l+1)*np.log(1/p)))

    L = np.linspace(0, L0*2, 100)
    y0 = L*delta*x
    y1 = p**(L*(1+l))*np.sqrt(8*n)*np.ones(100)
    y2 = delta*x*(L0+np.sqrt(2*n)*(p**(L0*(l+1))-p**(L*(l+1)))/(1-p**(l+1)))

    axis.plot(L, y0, label="Noiseless", color=colors[0])
    axis.plot(L, y1, label="Noisy", color=colors[1])
    axis.plot(L, y2, label="Noisy2", color=colors[2])

    axis.set_xlabel("L")
    axis.set_ylabel("Tr[rho(x)-rho(x')]")
    axis.set_ylim(0,2)
    # axis.set_title(f"")                
    plt.legend(loc="lower right")
    plt.show()

#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#%%

import numpy as np
import matplotlib.pyplot as plt
import rsmf
import matplotlib

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']

formatter = rsmf.setup(r"\documentclass[a4paper,twocolumn,notitlepage,nofootinbib]{revtex4-2}")

fig = formatter.figure(aspect_ratio=0.6)

plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

x = 0.005
delta = 1

n = 1000
l = 2
p = 0.9
L0 = int(1+np.log(2*n)/(2*(l+1)*np.log(1/p)))

L = np.arange(0, 50)
y0 = L*delta*x
y1 = p**(L*(1+l))*np.sqrt(8*n)
y2 = np.zeros_like(y1)
y2[:L0] = y0[:L0]
y2[L0:] = delta*x*(L0+np.sqrt(2*n)*(p**(L0*(l+1))-p**(L[L0:]*(l+1)))/(1-p**(l+1)))

_LW = 1.2
plt.plot(L, y0, label="Continuity bound", color="C0", lw=_LW)
plt.plot(L, y1, label="Contraction bound", color="C1", lw=_LW)
plt.plot(L, y2, label="Combined bound", color="C2", lw=_LW)

plt.xlabel(r"L")
plt.ylabel(r"Distance bound")
plt.ylim(0,.3)
plt.gca().set_yticks([0,0.1,0.2,0.3])
plt.xlim(0, 50)

plt.legend(loc="upper right", prop={"size": formatter.fontsizes.footnotesize})
plt.grid()
plt.tight_layout()
plt.savefig("example.pdf", bbox_inches="tight")

# plt.gca for axis
# %%
