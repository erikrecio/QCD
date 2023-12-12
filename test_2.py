# https://pennylane.ai/qml/demos/tutorial_variational_classifier/

#%%
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt

dataset = 2 #1-Pennylane, 2-Circles
parallel = False

num_qubits = 4 #2
num_layers_encoding = 3 #3
num_layers_unitaries = 6
num_iters = 60

one_shot = 0 # 0 - nothing, 1 - probabilities squared, 2 - variance
strength = 1/2 # lambda of the regularisation


dev = qml.device("default.qubit", wires=num_qubits)

def circles_dataset(num_datapoints, R, square_size):
    # num_datapoints = 100
    # R = 0.2 #radius circumference
    # square_size = 1

    np.random.seed(0)
    r = np.random.rand(num_datapoints)*R
    
    theta = np.random.rand(num_datapoints)*2*np.pi
    circle = np.array(list(zip(np.sqrt(r)*np.cos(theta),np.sqrt(r)*np.sin(theta))))

    c1 = circle/2+np.array([R,square_size-R])
    c2 = circle/2+np.array([square_size-R,R])

    c1 = np.append(c1, [[1]]*num_datapoints, axis=1)
    c2 = np.append(c2, [[-1]]*num_datapoints, axis=1)

    data = np.append(c1,c2,axis=0)

    return data


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


if dataset == 1:
    data = np.loadtxt("data/iris_classes1and2_scaled.txt")

elif dataset == 2:
    num_datapoints = 50 #100
    R = 0.2 #radius circumference
    square_size = 1
    data = circles_dataset(num_datapoints, R, square_size)


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
bias = []
w = weights_init
b = bias_init
for it in range(num_iters): #60

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    w, b, _, _ = opt.step(cost, w, b, feats_train_batch, Y_train_batch)
    weights.append(w)
    bias.append(b)
    
    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier(w, b, f)) for f in feats_train]
    predictions_val = [np.sign(variational_classifier(w, b, f)) for f in feats_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f}"
        "".format(it + 1, cost(w, b, features, Y), acc_train, acc_val)
    )
    
    # if acc_train == 1:
    #     break

#%%

plot_iter = 100 #it+1

plt.figure()
cm = plt.cm.RdBu

# make data for decision regions
xx, yy = np.meshgrid(np.linspace(-1.1, 1.1, 20), np.linspace(-1.1, 1.1, 20))
X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

# preprocess grid points like data inputs above
padding = 0.3 * np.ones((len(X_grid), 1))
X_grid = np.c_[np.c_[X_grid, padding], np.zeros((len(X_grid), 1))]  # pad each input
normalization = np.sqrt(np.sum(X_grid ** 2, -1))
X_grid = (X_grid.T / normalization).T  # normalize each input
features_grid = np.array(
    [get_angles(x) for x in X_grid]
)  # angles for state preparation are new features
predictions_grid = [variational_classifier(weights[plot_iter-1], bias[plot_iter-1], f) for f in features_grid]
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
plt.axis('square')
plt.show()


#%%

import pandas as pd
from datetime import datetime
from datetime import datetime
import pytz


it_1 = 60
it_2 = 100

predictions_1 = [variational_classifier(weights[it_1-1], bias[it_1-1], f) for f in features]
predictions_2 = [variational_classifier(weights[it_2-1], bias[it_2-1], f) for f in features]

z = list(zip(X, Y, predictions_1, predictions_2))


data = []
for x, y, p1, p2 in z:
    
    if dataset == 1:
        x = x.numpy()
        y= y.item()

    p1 = p1.numpy()
    p2 = p2.numpy()
    
    conf_1 = round((p1+1)/2, 4) if y==1 else round(1-(p1+1)/2, 4)
    conf_2 = round((p2+1)/2, 4) if y==1 else round(1-(p2+1)/2, 4)

    guess_1 = "Ok" if np.sign(p1)*y == 1 else "Error"
    guess_2 = "Ok" if np.sign(p2)*y == 1 else "Error"
    
    increment = (conf_2-conf_1)/conf_1
    
    row = {}
    row["X"], row["Y"], row[f"conf_{it_1}"], row[f"guess_{it_1}"], row[f"conf_{it_2}"], row[f"guess_{it_2}"], row["increment"] = x, y, conf_1, guess_1, conf_2, guess_2, increment

    data.append(row)

data = pd.DataFrame(data)
time_now = datetime.now(pytz.timezone('Europe/Andorra')).strftime("%Y-%m-%d %H-%M-%S")
file_name = f'{time_now} - q={num_qubits}, e={num_layers_encoding}, u={num_layers_unitaries}, str={strength}'
data.to_excel(f'results/{file_name}.xlsx', index=False)





#%%

one_shot = 2 # 0 - nothing, 1 - probabilities squared, 2 - variance
strength = 1/2 # lambda of the regularisation - default = 1/2

# continue the variational classifier
for it in range(60, 100):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    w, b, _, _ = opt.step(cost, w, b, feats_train_batch, Y_train_batch)
    weights.append(w)
    bias.append(b)
    
    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier(w, b, f)) for f in feats_train]
    predictions_val = [np.sign(variational_classifier(w, b, f)) for f in feats_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f}"
        "".format(it + 1, cost(w, b, features, Y), acc_train, acc_val)
    )
    

#%%

weights = weights[:60]
bias = bias[:60]


#%%

def circles_dataset(num_datapoints, R, square_size):
    # num_datapoints = 100
    # R = 0.2 #radius circumference
    # square_size = 1

    np.random.seed(0)
    r = np.random.rand(num_datapoints)*R
    print(r)
    theta = np.random.rand(num_datapoints)*2*np.pi
    circle = np.array(list(zip(np.sqrt(r)*np.cos(theta),np.sqrt(r)*np.sin(theta))))

    c1 = circle/2+np.array([R,square_size-R])
    c2 = circle/2+np.array([square_size-R,R])

    c1 = np.append(c1, [[1]]*num_datapoints, axis=1)
    c2 = np.append(c2, [[-1]]*num_datapoints, axis=1)

    data = np.append(c1,c2,axis=0)

    return data

num_datapoints = 100
R = 0.2 #radius circumference
square_size = 1
circles_dataset(num_datapoints, R, square_size)


#%%

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    
    if one_shot == 1:
        regularisation = -sum([((p+1)/2)**2 + (1-(p+1)/2)**2 for p in predictions])
    elif one_shot == 2:
        regularisation = sum([(1 - p**2)**2 for p in predictions])
    else:
        regularisation = 0
    
    
    loss = (loss + strength*regularisation)/ len(labels)

    return loss
