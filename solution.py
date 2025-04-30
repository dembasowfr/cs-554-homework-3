import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

X_train = train_df["x"].values.reshape(-1, 1)
y_train = train_df["r"].values
X_test = test_df["x"].values.reshape(-1, 1)
y_test = test_df["r"].values

# Add bias term
def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

X_train_b = add_bias(X_train)
X_test_b = add_bias(X_test)

# Activation functions
def tanh(x): return np.tanh(x)
def tanh_deriv(z): return 1 - z ** 2



# ---- SINGLE-LAYER ----
def train_single_layer(X, y, w, epochs, lr):
    mse_list = []
    for epoch in range(epochs):
        y_pred = X @ w
        error = y - y_pred
        grad = -X.T @ error / len(X)
        w -= lr * grad
        mse = np.mean(error**2)
        mse_list.append(mse)
    return w, mse_list

# ---- MULTI-LAYER ----
def train_mlp(X, y, n_hidden, epochs, lr):
    n_input = X.shape[1]
    np.random.seed(42)
    W = np.random.randn(n_hidden, n_input) * 0.1
    V = np.random.randn(1, n_hidden + 1) * 0.1

    mse_list = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            x = X[i].reshape(-1, 1)
            r = y[i]

            z = tanh(W @ x)
            z_b = np.vstack([z, [[1]]])
            y_pred = V @ z_b

            error = r - y_pred.item()
            total_error += error**2

            dV = lr * error * z_b.T
            #dz = tanh_deriv(z) * (V[:, :-1].T @ error)
            dz = tanh_deriv(z) * (V[:, :-1].T * error)
            dW = lr * dz @ x.T

            V += dV
            W += dW

        mse = total_error / len(X)
        mse_list.append(mse)

    return W, V, mse_list

# ---- PREDICTION ----
def predict_single(X, w):
    return (X @ w).flatten()

def predict_mlp(X, W, V):
    Z = tanh(X @ W.T)
    Z_b = np.hstack([Z, np.ones((Z.shape[0], 1))])
    return (Z_b @ V.T).flatten()

# ---- TRAIN ALL MODELS ----

# Adjust learning rate and epochs
# lr = 0.001  # Further reduced learning rate for finer updates
# epochs = 1000  # Reduced epochs to save computation time

# lr = 0.005  # Further reduced learning rate for finer updates
# epochs = 1500  # Reduced epochs to save computation time


lr = 0.002  # Further reduced learning rate for finer updates
epochs = 1200  # Reduced epochs to save computation time
# lr = 0.002  # Further reduced learning rate for finer updates
# epochs = 10000  # Reduced epochs to save computation time

# 1. Single-layer
w = np.random.randn(2) * 0.01
w, mse_0 = train_single_layer(X_train_b, y_train, w, epochs, lr)
ypred_0 = predict_single(X_train_b, w)

# 2. MLP with 2 hidden units
W2, V2, mse_2 = train_mlp(X_train_b, y_train, 2, epochs, lr)
ypred_2 = predict_mlp(X_train_b, W2, V2)

# 3. MLP with 4 hidden units
W4, V4, mse_4 = train_mlp(X_train_b, y_train, 4, epochs, lr)
ypred_4 = predict_mlp(X_train_b, W4, V4)

# 4. MLP with 8 hidden units
W8, V8, mse_8 = train_mlp(X_train_b, y_train, 8, epochs, lr)
ypred_8 = predict_mlp(X_train_b, W8, V8)

# Calculate final MSE values
mse_train_single = np.mean((y_train - ypred_0)**2)
mse_test_single = np.mean((y_test - predict_single(X_test_b, w))**2)

mse_train_2 = np.mean((y_train - ypred_2)**2)
mse_test_2 = np.mean((y_test - predict_mlp(X_test_b, W2, V2))**2)

mse_train_4 = np.mean((y_train - ypred_4)**2)
mse_test_4 = np.mean((y_test - predict_mlp(X_test_b, W4, V4))**2)

mse_train_8 = np.mean((y_train - ypred_8)**2)
mse_test_8 = np.mean((y_test - predict_mlp(X_test_b, W8, V8))**2)

# ---- PLOT ALL IN ONE WINDOW ----
fig, axs = plt.subplots(4, 2, figsize=(14, 14))
titles = [
    "Single-layer Perceptron",
    "MLP with 2 Hidden Units",
    "MLP with 4 Hidden Units",
    "MLP with 8 Hidden Units"
]
x_line = np.linspace(min(X_train)[0], max(X_train)[0], 200).reshape(-1, 1)
x_line_b = add_bias(x_line)

for i, (name, y_pred_train, mse, model_type) in enumerate([
    (titles[0], ypred_0, mse_0, 'single'),
    (titles[1], ypred_2, mse_2, 'mlp2'),
    (titles[2], ypred_4, mse_4, 'mlp4'),
    (titles[3], ypred_8, mse_8, 'mlp8'),
]):
    axs[i, 0].scatter(X_train, y_train, label='Training Data', color='blue')
    
    if model_type == 'single':
        y_line = predict_single(x_line_b, w)
    elif model_type == 'mlp2':
        y_line = predict_mlp(x_line_b, W2, V2)
    elif model_type == 'mlp4':
        y_line = predict_mlp(x_line_b, W4, V4)
    elif model_type == 'mlp8':
        y_line = predict_mlp(x_line_b, W8, V8)

    axs[i, 0].plot(x_line, y_line, color='red', label='Model Output')
    axs[i, 0].set_title(f'{name}: Output vs Training Data')
    axs[i, 0].set_xlabel('x')
    axs[i, 0].set_ylabel('r')
    axs[i, 0].legend()
    axs[i, 0].grid()

    axs[i, 1].plot(mse, color='green')
    axs[i, 1].set_title(f'{name}: MSE over Epochs')
    axs[i, 1].set_xlabel('Epochs')
    axs[i, 1].set_ylabel('MSE')
    axs[i, 1].grid()

plt.tight_layout()
plt.savefig(f'output/models_lr-{lr}-epochs-{epochs}.png')

plt.show()

# Display the outputs after training
print("\n--- Final Outputs ---")
print("Single-layer Perceptron: Train MSE =", mse_train_single, ", Test MSE =", mse_test_single)
print("MLP with 2 Hidden Units: Train MSE =", mse_train_2, ", Test MSE =", mse_test_2)
print("MLP with 4 Hidden Units: Train MSE =", mse_train_4, ", Test MSE =", mse_test_4)
print("MLP with 8 Hidden Units: Train MSE =", mse_train_8, ", Test MSE =", mse_test_8)

# Generate and save plots for each model
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), mse_0, label='Single-layer Perceptron', color='blue')
plt.plot(range(epochs), mse_2, label='MLP (2 Hidden Units)', color='green')
plt.plot(range(epochs), mse_4, label='MLP (4 Hidden Units)', color='orange')
plt.plot(range(epochs), mse_8, label='MLP (8 Hidden Units)', color='red')
plt.title('MSE vs Epochs for All Models')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid()
plt.savefig(f'output/mse_lr_{lr}_epochs_{epochs}.png')
plt.show()