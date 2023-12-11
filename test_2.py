# https://pennylane.ai/qml/demos/tutorial_variational_classifier/

#%%
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

num_qubits = 2
num_layers_encoding = 3
num_layers_unitaries = 6
num_iters = 100
one_shot = 2 # 0 - nothing, 1 - probabilities squared, 2 - variance
strength = 1/2 # lambda of the regularisation


dev = qml.device("default.qubit", wires=num_qubits)

def get_angles(x):

    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def statepreparation(a):
    
    for i in range(num_qubits):
        qml.RY(a[0], wires=i)
        if i != num_qubits-1:
            qml.CNOT(wires=[i, i+1])
        qml.RY(a[1], wires=i)
        if i != num_qubits-1:
            qml.CNOT(wires=[i, i+1])
        qml.RY(a[2], wires=i)
        if i != num_qubits-1:
            qml.CNOT(wires=[i, i+1])
        
        qml.PauliX(wires=i)
        
        qml.RY(a[3], wires=i)
        if i != num_qubits-1:
            qml.CNOT(wires=[i, i+1])
        qml.RY(a[4], wires=i)
        if i != num_qubits-1:
            qml.CNOT(wires=[i, i+1])

        qml.PauliX(wires=i)
        
    # qml.RY(a[0], wires=0)

    # qml.CNOT(wires=[0, 1])
    # qml.RY(a[1], wires=1)
    # qml.CNOT(wires=[0, 1])
    # qml.RY(a[2], wires=1)

    # qml.PauliX(wires=0)
    # qml.CNOT(wires=[0, 1])
    # qml.RY(a[3], wires=1)
    # qml.CNOT(wires=[0, 1])
    # qml.RY(a[4], wires=1)
    # qml.PauliX(wires=0)


def layer(W):
    
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        # if i != num_qubits-1:
        #     qml.CNOT([i,i+1])
            
    for i in range(num_qubits-1):
        qml.CNOT([i,i+1])
    
    # qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    # qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    # qml.CNOT(wires=[0, 1])
    
    
@qml.qnode(dev, interface="autograd")
def circuit(weights, angles):
    
    for j in range(num_layers_unitaries):
        layer(weights[j])
    
    for i in range(num_layers_encoding):
        
        statepreparation(angles)
        
        for j in range(num_layers_unitaries):
            layer(weights[(i+1)*num_layers_unitaries + j])


    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, angles):
    return circuit(weights, angles) + bias

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    
    if one_shot == 1:
        regularisation = -sum([((p+1)/2)**2 + (1-(p+1)/2)**2 for p in predictions])
    elif one_shot == 2:
        regularisation = sum([1 - p**2 for p in predictions])
    else:
        regularisation = 0
    
    
    loss = (loss + strength*regularisation)/ len(labels)

    return loss

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

def cost(weights, bias, features, labels):
    predictions = [variational_classifier(weights, bias, f) for f in features]
    return square_loss(labels, predictions)

data = np.loadtxt("data/iris_classes1and2_scaled.txt")
X = data[:, 0:2]
print("First X sample (original)  :", X[0])

# pad the vectors to size 2^2 with constant values
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
print("First X sample (padded)    :", X_pad[0])

# normalize each input
normalization = np.sqrt(np.sum(X_pad ** 2, -1))
X_norm = (X_pad.T / normalization).T
print("First X sample (normalized):", X_norm[0])

# angles for state preparation are new features
features = np.array([get_angles(x) for x in X_norm], requires_grad=False)
print("First features sample      :", features[0])

Y = data[:, -1]


import matplotlib.pyplot as plt


# plt.figure()
# plt.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c="b", marker="o", edgecolors="k")
# plt.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], c="r", marker="o", edgecolors="k")
# plt.title("Original data")
# plt.show()

# plt.figure()
# dim1 = 0
# dim2 = 1
# plt.scatter(
#     X_norm[:, dim1][Y == 1], X_norm[:, dim2][Y == 1], c="b", marker="o", edgecolors="k"
# )
# plt.scatter(
#     X_norm[:, dim1][Y == -1], X_norm[:, dim2][Y == -1], c="r", marker="o", edgecolors="k"
# )
# plt.title("Padded and normalised data (dims {} and {})".format(dim1, dim2))
# plt.show()

# plt.figure()
# dim1 = 0
# dim2 = 3
# plt.scatter(
#     features[:, dim1][Y == 1], features[:, dim2][Y == 1], c="b", marker="o", edgecolors="k"
# )
# plt.scatter(
#     features[:, dim1][Y == -1], features[:, dim2][Y == -1], c="r", marker="o", edgecolors="k"
# )
# plt.title("Feature vectors (dims {} and {})".format(dim1, dim2))
# plt.show()

np.random.seed(0)
num_data = len(Y)
num_train = int(0.75 * num_data)
index = np.random.permutation(range(num_data))
feats_train = features[index[:num_train]]
Y_train = Y[index[:num_train]]
feats_val = features[index[num_train:]]
Y_val = Y[index[num_train:]]

# We need these later for plotting
X_train = X[index[:num_train]]
X_val = X[index[num_train:]]

num_layers = (num_layers_encoding + 1)*num_layers_unitaries
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

opt = NesterovMomentumOptimizer(0.01)
batch_size = 5

# train the variational classifier
weights = []
w = weights_init
bias = bias_init
for it in range(num_iters): #60

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    w, bias, _, _ = opt.step(cost, w, bias, feats_train_batch, Y_train_batch)
    weights.append(w)
    
    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier(w, bias, f)) for f in feats_train]
    predictions_val = [np.sign(variational_classifier(w, bias, f)) for f in feats_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, cost(w, bias, features, Y), acc_train, acc_val)
    )
    
#%%

plot_iter = 60

plt.figure()
cm = plt.cm.RdBu

# make data for decision regions
xx, yy = np.meshgrid(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))
X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

# preprocess grid points like data inputs above
padding = 0.3 * np.ones((len(X_grid), 1))
X_grid = np.c_[np.c_[X_grid, padding], np.zeros((len(X_grid), 1))]  # pad each input
normalization = np.sqrt(np.sum(X_grid ** 2, -1))
X_grid = (X_grid.T / normalization).T  # normalize each input
features_grid = np.array(
    [get_angles(x) for x in X_grid]
)  # angles for state preparation are new features
predictions_grid = [variational_classifier(weights[plot_iter-1], bias, f) for f in features_grid]
Z = np.reshape(predictions_grid, xx.shape)

# plot decision regions
cnt = plt.contourf(
    xx, yy, Z, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=0.8, extend="both"
)
plt.contour(
    xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,)
)
plt.colorbar(cnt, ticks=[-1, 0, 1])

# plot data
plt.scatter(
    X_train[:, 0][Y_train == 1],
    X_train[:, 1][Y_train == 1],
    c="b",
    marker="o",
    edgecolors="k",
    label="class 1 train",
)
plt.scatter(
    X_val[:, 0][Y_val == 1],
    X_val[:, 1][Y_val == 1],
    c="b",
    marker="^",
    edgecolors="k",
    label="class 1 validation",
)
plt.scatter(
    X_train[:, 0][Y_train == -1],
    X_train[:, 1][Y_train == -1],
    c="r",
    marker="o",
    edgecolors="k",
    label="class -1 train",
)
plt.scatter(
    X_val[:, 0][Y_val == -1],
    X_val[:, 1][Y_val == -1],
    c="r",
    marker="^",
    edgecolors="k",
    label="class -1 validation",
)

plt.legend()
plt.show()


#%%

import pandas as pd
from datetime import datetime
from datetime import datetime
import pytz

predictions_train = [variational_classifier(weights[plot_iter-1], bias, f) for f in feats_train]
# predictions_val = [variational_classifier(w, bias, f) for f in feats_val]
z = list(zip(X_train, predictions_train, Y_train))


data = []
for xt, p, yt in z:

    pred = round((p.numpy()+1)/2, 4)

    if np.sign(p.numpy())*yt == 1:
        guess = "Ok"
    else:
        guess = "Error"
    
    row = {}
    row["X"], row["Y"], row["pred"], row["guess"] = xt.numpy(), yt.item(), pred, guess

    data.append(row)

data = pd.DataFrame(data)
time_now = datetime.now(pytz.timezone('Europe/Andorra')).strftime("%Y-%m-%d %H-%M-%S")
file_name = f'{time_now} - regularization, one_shot = {one_shot}'
data.to_csv(f'results/{file_name}.csv', index=False)

