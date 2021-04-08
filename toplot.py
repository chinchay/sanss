#%%
import matplotlib.pyplot as plt
import csv
import numpy as np

#%%
with open("upToMoves.csv",newline="") as csvfile:
    upToMoves = list(csv.reader(csvfile))
    upToMoves = int(upToMoves[0][0])
#
with open("namePlot.txt", "r") as f:
    namePlot = f.read()
#
# print("python: ", namePlot)
#%%
with open("factNumPoints.csv",newline="") as csvfile:
    factNumPoints = list(csv.reader(csvfile))
    factNumPoints = int(factNumPoints[0][0])
#%%
with open("w_E.csv",newline="") as csvfile:
    w_E = np.array(list(csv.reader(csvfile)))
    w_E = w_E.astype(np.float)
#
a, b = w_E.shape
w_E = w_E.reshape(b, a)
nWalkers = b

#%%
with open("shouldIplotT.csv",newline="") as csvfile:
    shouldIplotT = list(csv.reader(csvfile))
    shouldIplotT = (shouldIplotT == "true")
#

if shouldIplotT:
    with open("record_T.csv",newline="") as csvfile:
        record_T = np.array(list(csv.reader(csvfile)))
        record_T = record_T.astype(np.float)
    #
    a, b = record_T.shape
    record_T = record_T.reshape(b, a)
    record_T = record_T[0]
#

#%%
#%%
fig, ax1 = plt.subplots()

myColors = ["k", "r", "g", "m", "y"]
# print(w_E)


for w in range(nWalkers):
    # y = [ w_E[w][i] for i in 1:length(w_E[w]) if i % factNumPoints == 0 ]
    y = [ w_E[w, i] for i in range(upToMoves) if i % factNumPoints == 0]
    x = [i for i in range(len(y))]
    if nWalkers <= 5:
        ax1.scatter(x, y, s=0.5, color=myColors[w])
    else:
        ax1.scatter(x, y, s=0.5)
    #
#
ax1.set_ylabel("energy (eV)", color="k")
ax1.set_ylim([-5600, -5150])
if factNumPoints == 1:
    xlabel = "move"
else:
    xlabel = "move x " + str(factNumPoints)
#
ax1.set_xlabel(xlabel)
#
# if record_T !== nothing

xT = []
yT = []


# shouldIplotT = false
if shouldIplotT:
    if len(record_T) >= 1:
        yT = [ record_T[i] for i in range(upToMoves) if i % factNumPoints == 0 ]
        xT = [i for i in range(len(yT))]
    #
    ax2 = ax1.twinx()
    ax2.plot(xT, yT, "b", linewidth=0.4)
    ax2.set_ylabel("effective temperature (a.u.)", color="b")
    ax2.tick_params(axis="y", colors="blue")
#

fig.savefig(namePlot)
plt.close(fig)




# %%
